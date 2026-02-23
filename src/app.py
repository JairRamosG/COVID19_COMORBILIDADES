import streamlit as st
from pathlib import Path
import os
from pyspark.sql import SparkSession


#############################
import streamlit as st
from pyspark.sql import functions as F
import plotly.express as px #
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
#############################

st.set_page_config(
    page_title="Dashboard COVID-19 Comorbilidades",
    page_icon="",
    layout="wide")


st.title("🦠 Dashboard Interactivo de COVID-19 en México")
st.markdown("""
Este dashboard permite explorar los datos de pacientes COVID-19 en México, 
analizando comorbilidades, distribución por edad, género y resultados.
""")
st.markdown("---")

# Iniciar la sesion de Spark 
@st.cache_resource
def init_spark():
    '''
    INicio de la sesión de Spark para poder leer los datos
    '''

    spark = SparkSession.builder \
        .appName('comorbilidades')\
        .config("spark.sql.adaptive.enabled", "true")\
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")\
        .config("spark.driver.memory", "4g")\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()
    
    return spark

@st.cache_resource
def cargar_datos_spark():
    '''
    Carga el dataFrame de Spark de un parquet
    '''
    spark = init_spark()
    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_PARQUET_PATH = Path(os.getenv('DATA_PARQUET', BASE_DIR / 'data' / 'parquet' / 'df_final.parquet'))

    if not DATA_PARQUET_PATH.exists():
        st.error(f'No se encuentra el archivo en la ruta: {DATA_PARQUET_PATH}')
        return None

    df_spark = spark.read.parquet(str(DATA_PARQUET_PATH))
    
    return df_spark
#########################################################################################################################


# ============================================
# FUNCIONES AUXILIARES
# ============================================
def obtener_metricas(df_spark):
    """Calcula métricas agregadas"""
    return df_spark.agg(
        F.count('*').alias('total'),
        F.avg('EDAD').alias('edad_media'),
        F.sum('SOBREVIVIO').alias('sobrevivieron'),
        F.sum(F.when(F.col('SOBREVIVIO') == 0, 1).otherwise(0)).alias('fallecidos'),
        F.avg('N_COMORBILIDADES').alias('comorbilidades_promedio')
    ).collect()[0]

def filtrar_por_edad(df_spark, edad_min, edad_max):
    """Filtra por rango de edad"""
    return df_spark.filter(
        (F.col('EDAD') >= edad_min) & 
        (F.col('EDAD') <= edad_max)
    )

def filtrar_por_categoria_edad(df_spark, categorias):
    """Filtra por categorías de edad seleccionadas"""
    if categorias and 'Todas' not in categorias:
        return df_spark.filter(F.col('CATEGORIA_EDAD').isin(categorias))
    return df_spark

def filtrar_por_sexo(df_spark, sexos):
    """Filtra por sexo seleccionado"""
    if sexos and 'Todos' not in sexos:
        # Convertir texto a número (1 = Mujer, 2 = Hombre según tus datos)
        sexo_map = {'Mujer': 1, 'Hombre': 2}
        sexos_num = [sexo_map[s] for s in sexos if s in sexo_map]
        if sexos_num:
            return df_spark.filter(F.col('SEXO').isin(sexos_num))
    return df_spark

def filtrar_por_sobrevivencia(df_spark, opcion):
    """Filtra por estado de sobrevivencia"""
    if opcion == 'Sobrevivieron':
        return df_spark.filter(F.col('SOBREVIVIO') == 1)
    elif opcion == 'Fallecidos':
        return df_spark.filter(F.col('SOBREVIVIO') == 0)
    return df_spark  # Todos

def filtrar_por_comorbilidades(df_spark, comorbilidades, tipo_filtro):
    """Filtra por comorbilidades"""
    if not comorbilidades:
        return df_spark
    
    if tipo_filtro == "CON alguna":
        # OR - al menos una
        condicion = F.lit(False)
        for c in comorbilidades:
            if c in df_spark.columns:
                condicion = condicion | (F.col(c) == 1)
        return df_spark.filter(condicion)
    
    elif tipo_filtro == "TODAS":
        # AND - todas las seleccionadas
        for c in comorbilidades:
            if c in df_spark.columns:
                df_spark = df_spark.filter(F.col(c) == 1)
        return df_spark
    
    else:  # "SIN ninguna"
        # NOT - ninguna de las seleccionadas
        condicion = F.lit(True)
        for c in comorbilidades:
            if c in df_spark.columns:
                condicion = condicion & (F.col(c) != 1)
        return df_spark.filter(condicion)

def filtrar_por_riesgo(df_spark, riesgos):
    """Filtra por nivel de riesgo"""
    if riesgos and 'Todos' not in riesgos:
        return df_spark.filter(F.col('RIESGO').isin(riesgos))
    return df_spark

# ============================================
# SIDEBAR - FILTROS
# ============================================
st.sidebar.header("🔍 Filtros")

# Cargar datos
with st.spinner("🔄 Cargando datos..."):
    df_spark = cargar_datos_spark()
    
    # Obtener valores para filtros
    edad_min, edad_max = df_spark.select(F.min('EDAD'), F.max('EDAD')).collect()[0]
    categorias_edad = ['Todas'] + [r[0] for r in df_spark.select('CATEGORIA_EDAD').distinct().collect()]
    niveles_riesgo = ['Todos'] + [r[0] for r in df_spark.select('RIESGO').distinct().collect()]

# ============================================
# FILTROS INTERACTIVOS
# ============================================
with st.sidebar.expander("📊 Filtros básicos", expanded=True):
    # Rango de edad
    rango_edad = st.slider(
        "Rango de edad",
        min_value=int(edad_min),
        max_value=int(edad_max),
        value=(int(edad_min), int(edad_max))
    )
    
    # Categoría de edad
    categorias_sel = st.multiselect(
        "Categoría de edad",
        options=categorias_edad,
        default=['Todas']
    )
    
    # Sexo
    sexo_sel = st.multiselect(
        "Sexo",
        options=['Todos', 'Mujer', 'Hombre'],
        default=['Todos']
    )
    
    # Sobrevivencia
    sobrevivencia_sel = st.radio(
        "Estado",
        options=['Todos', 'Sobrevivieron', 'Fallecidos'],
        index=0,
        horizontal=True
    )

with st.sidebar.expander("🫀 Comorbilidades", expanded=True):
    # Lista de comorbilidades disponibles en tus datos
    comorbilidades = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 
                      'EPOC', 'INMUSUPR', 'RENAL_CRONICA', 'TABAQUISMO',
                      'CARDIOVASCULAR', 'OTRA_COM']
    
    tipo_filtro_comorb = st.radio(
        "Tipo de filtro",
        options=["CON alguna", "TODAS", "SIN ninguna"],
        index=0,
        help="CON alguna: pacientes con al menos una\nTODAS: pacientes con todas\nSIN ninguna: pacientes sin ninguna"
    )
    
    comorb_sel = st.multiselect(
        "Selecciona comorbilidades",
        options=comorbilidades,
        default=[]
    )

with st.sidebar.expander("⚕️ Nivel de Riesgo", expanded=False):
    riesgo_sel = st.multiselect(
        "Nivel de riesgo",
        options=niveles_riesgo,
        default=['Todos']
    )

# ============================================
# APLICAR FILTROS
# ============================================
df_filtrado = df_spark

# Aplicar filtros secuencialmente
df_filtrado = filtrar_por_edad(df_filtrado, rango_edad[0], rango_edad[1])
df_filtrado = filtrar_por_categoria_edad(df_filtrado, categorias_sel)
df_filtrado = filtrar_por_sexo(df_filtrado, sexo_sel)
df_filtrado = filtrar_por_sobrevivencia(df_filtrado, sobrevivencia_sel)
df_filtrado = filtrar_por_riesgo(df_filtrado, riesgo_sel)
df_filtrado = filtrar_por_comorbilidades(df_filtrado, comorb_sel, tipo_filtro_comorb)

# Obtener métricas
metricas = obtener_metricas(df_filtrado)

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================
st.header("📊 Resumen General")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total pacientes",
        f"{metricas['total']:,}"
    )

with col2:
    st.metric(
        "Edad media",
        f"{metricas['edad_media']:.1f} años"
    )

with col3:
    st.metric(
        "Sobrevivieron",
        f"{metricas['sobrevivieron']:,}",
        delta=f"{metricas['sobrevivieron']/metricas['total']*100:.1f}%"
    )

with col4:
    st.metric(
        "Fallecidos",
        f"{metricas['fallecidos']:,}",
        delta=f"{metricas['fallecidos']/metricas['total']*100:.1f}%"
    )

with col5:
    st.metric(
        "Comorbilidades promedio",
        f"{metricas['comorbilidades_promedio']:.2f}"
    )

st.markdown("---")

# ============================================
# GRÁFICO 1: Distribución por edad y resultado
# ============================================
st.subheader("📈 Distribución por Edad y Resultado")

# Tomar muestra para gráficos (10,000 filas es suficiente)
muestra_pd = df_filtrado.select('EDAD', 'SOBREVIVIO').sample(0.1).limit(10000).toPandas()
muestra_pd['RESULTADO'] = muestra_pd['SOBREVIVIO'].map({1: 'Sobrevivió', 0: 'Falleció'})

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        muestra_pd,
        x='EDAD',
        nbins=50,
        color='RESULTADO',
        title="Distribución de edades",
        color_discrete_map={'Sobrevivió': '#2ecc71', 'Falleció': '#e74c3c'},
        barmode='overlay'
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(
        muestra_pd,
        x='RESULTADO',
        y='EDAD',
        title="Edad por resultado",
        color='RESULTADO',
        color_discrete_map={'Sobrevivió': '#2ecc71', 'Falleció': '#e74c3c'}
    )
    fig_box.update_layout(height=400)
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================
# GRÁFICO 2: Análisis de Comorbilidades
# ============================================
st.subheader("🫀 Análisis de Comorbilidades")

col1, col2 = st.columns(2)

with col1:
    # Frecuencia de comorbilidades
    comorb_counts = []
    for c in comorbilidades:
        if c in df_filtrado.columns:
            count = df_filtrado.filter(F.col(c) == 1).count()
            pct = (count / metricas['total'] * 100) if metricas['total'] > 0 else 0
            comorb_counts.append({
                'Comorbilidad': c,
                'Conteo': count,
                'Porcentaje': pct
            })
    
    df_comorb = pd.DataFrame(comorb_counts).sort_values('Conteo', ascending=False)
    
    if not df_comorb.empty:
        fig_comorb = px.bar(
            df_comorb,
            x='Comorbilidad',
            y='Conteo',
            title="Frecuencia de comorbilidades",
            color='Porcentaje',
            color_continuous_scale='Reds',
            text='Conteo'
        )
        fig_comorb.update_traces(textposition='outside')
        fig_comorb.update_layout(height=500)
        st.plotly_chart(fig_comorb, use_container_width=True)

with col2:
    # Número de comorbilidades por paciente
    comorb_dist = df_filtrado.groupBy('N_COMORBILIDADES').count().orderBy('N_COMORBILIDADES').toPandas()
    
    if not comorb_dist.empty:
        fig_dist = px.bar(
            comorb_dist,
            x='N_COMORBILIDADES',
            y='count',
            title="Distribución del número de comorbilidades",
            color='count',
            color_continuous_scale='Viridis'
        )
        fig_dist.update_layout(
            xaxis_title="Número de comorbilidades",
            yaxis_title="Cantidad de pacientes",
            height=500
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ============================================
# GRÁFICO 3: Análisis por Categoría de Edad
# ============================================
st.subheader("📊 Análisis por Categoría de Edad")

# Crear tabla de mortalidad por categoría de edad
tabla_mortalidad = df_filtrado.groupBy('CATEGORIA_EDAD').agg(
    F.count('*').alias('total'),
    F.sum('SOBREVIVIO').alias('sobrevivieron'),
    F.sum(F.when(F.col('SOBREVIVIO') == 0, 1).otherwise(0)).alias('fallecidos')
).orderBy('CATEGORIA_EDAD').toPandas()

if not tabla_mortalidad.empty:
    tabla_mortalidad['tasa_mortalidad'] = (tabla_mortalidad['fallecidos'] / tabla_mortalidad['total'] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mortalidad = px.bar(
            tabla_mortalidad,
            x='CATEGORIA_EDAD',
            y='tasa_mortalidad',
            title="Tasa de mortalidad por categoría de edad",
            color='tasa_mortalidad',
            color_continuous_scale='Reds',
            text='tasa_mortalidad'
        )
        fig_mortalidad.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_mortalidad.update_layout(height=400)
        st.plotly_chart(fig_mortalidad, use_container_width=True)
    
    with col2:
        fig_stacked = px.bar(
            tabla_mortalidad.melt(id_vars=['CATEGORIA_EDAD'], value_vars=['sobrevivieron', 'fallecidos']),
            x='CATEGORIA_EDAD',
            y='value',
            color='variable',
            title="Sobrevivientes vs Fallecidos por categoría de edad",
            color_discrete_map={'sobrevivieron': '#2ecc71', 'fallecidos': '#e74c3c'},
            barmode='stack'
        )
        fig_stacked.update_layout(height=400)
        st.plotly_chart(fig_stacked, use_container_width=True)

# ============================================
# GRÁFICO 4: Análisis de Riesgo
# ============================================
st.subheader("⚕️ Análisis de Nivel de Riesgo")

riesgo_analysis = df_filtrado.groupBy('RIESGO').agg(
    F.count('*').alias('total'),
    F.avg('EDAD').alias('edad_promedio'),
    F.sum('SOBREVIVIO').alias('sobrevivieron'),
    F.sum(F.when(F.col('SOBREVIVIO') == 0, 1).otherwise(0)).alias('fallecidos')
).orderBy(F.when(F.col('RIESGO') == 'NORMAL', 1)
          .when(F.col('RIESGO') == 'MODERADO', 2)
          .when(F.col('RIESGO') == 'ALTO', 3)
          .otherwise(4)).toPandas()

if not riesgo_analysis.empty:
    riesgo_analysis['tasa_mortalidad'] = (riesgo_analysis['fallecidos'] / riesgo_analysis['total'] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_riesgo = px.pie(
            riesgo_analysis,
            values='total',
            names='RIESGO',
            title="Distribución por nivel de riesgo",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_riesgo.update_traces(textposition='inside', textinfo='percent+label')
        fig_riesgo.update_layout(height=400)
        st.plotly_chart(fig_riesgo, use_container_width=True)
    
    with col2:
        fig_tasa = px.bar(
            riesgo_analysis,
            x='RIESGO',
            y='tasa_mortalidad',
            title="Tasa de mortalidad por nivel de riesgo",
            color='tasa_mortalidad',
            color_continuous_scale='Reds',
            text='tasa_mortalidad'
        )
        fig_tasa.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_tasa.update_layout(height=400)
        st.plotly_chart(fig_tasa, use_container_width=True)

# ============================================
# TABLA DE DATOS
# ============================================
with st.expander("📋 Ver muestra de datos"):
    # Mostrar columnas más relevantes
    columnas_mostrar = ['EDAD', 'CATEGORIA_EDAD', 'SEXO', 'RIESGO', 'SOBREVIVIO',
                        'N_COMORBILIDADES', 'DIABETES', 'HIPERTENSION', 'OBESIDAD']
    
    # Filtrar solo columnas que existen
    columnas_existentes = [c for c in columnas_mostrar if c in df_filtrado.columns]
    
    muestra_df = df_filtrado.select(columnas_existentes).limit(1000).toPandas()
    
    # Mapear valores para mejor lectura
    if 'SEXO' in muestra_df.columns:
        muestra_df['SEXO'] = muestra_df['SEXO'].map({1: 'Mujer', 2: 'Hombre'})
    if 'SOBREVIVIO' in muestra_df.columns:
        muestra_df['SOBREVIVIO'] = muestra_df['SOBREVIVIO'].map({1: 'Sobrevivió', 0: 'Falleció'})
    
    st.dataframe(muestra_df, use_container_width=True)
    st.caption(f"Mostrando 1000 filas de {metricas['total']:,} totales")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📊 Total en base: {df_spark.count():,} registros")
with col2:
    st.caption(f"🔍 Mostrando: {metricas['total']:,} registros")
with col3:
    st.caption("🦠 Datos COVID-19 México - Análisis de Comorbilidades")