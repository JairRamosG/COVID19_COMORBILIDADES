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
@st.cache_resourse
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

@st.cache_resourse
def carga_datos_spark():
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
# FUNCIONES DE FILTRADO CON SPARK
# ============================================
def filtrar_datos_spark(df_spark, filtros):
    """Aplica filtros al DataFrame de Spark"""
    df_filtrado = df_spark
    
    # Filtro por sexo
    if filtros['sexo'] != 'Todos':
        df_filtrado = df_filtrado.filter(F.col('SEXO') == filtros['sexo'])
    
    # Filtro por edad
    df_filtrado = df_filtrado.filter(
        (F.col('EDAD') >= filtros['edad_min']) & 
        (F.col('EDAD') <= filtros['edad_max'])
    )
    
    # Filtro por resultado
    resultados_mostrar = []
    if filtros['mostrar_fallecidos']:
        resultados_mostrar.append('Fallecido')
    if filtros['mostrar_sobrevivientes']:
        resultados_mostrar.append('Sobreviviente')
    
    if resultados_mostrar:
        df_filtrado = df_filtrado.filter(F.col('RESULTADO').isin(resultados_mostrar))
    
    # Filtro por comorbilidades
    if filtros['comorbilidades']:
        if filtros['tipo_filtro'] == "CON alguna":
            # OR condition
            condicion = F.lit(False)
            for c in filtros['comorbilidades']:
                condicion = condicion | (F.col(c) == 1)
            df_filtrado = df_filtrado.filter(condicion)
        elif filtros['tipo_filtro'] == "SIN ninguna":
            # NOT condition (ninguna)
            condicion = F.lit(True)
            for c in filtros['comorbilidades']:
                condicion = condicion & (F.col(c) != 1)
            df_filtrado = df_filtrado.filter(condicion)
        else:  # AND (todas)
            for c in filtros['comorbilidades']:
                df_filtrado = df_filtrado.filter(F.col(c) == 1)
    
    return df_filtrado

def obtener_metricas(df_spark):
    """Calcula métricas agregadas con Spark"""
    return df_spark.agg(
        F.count('*').alias('total'),
        F.avg('EDAD').alias('edad_media'),
        F.sum(F.when(F.col('RESULTADO') == 'Fallecido', 1).otherwise(0)).alias('fallecidos'),
        F.sum(F.when(F.col('RESULTADO') == 'Sobreviviente', 1).otherwise(0)).alias('sobrevivientes')
    ).collect()[0]

def contar_comorbilidades(df_spark, comorbilidades):
    """Cuenta frecuencia de comorbilidades"""
    resultados = []
    for c in comorbilidades:
        if c in df_spark.columns:
            count = df_spark.filter(F.col(c) == 1).count()
            resultados.append((c, count))
    return resultados

# ============================================
# SIDEBAR - FILTROS
# ============================================
st.sidebar.header("🔍 Filtros")

# Obtener valores únicos para filtros (con Spark)
@st.cache_data
def obtener_valores_unicos(df_spark, columna):
    return [row[0] for row in df_spark.select(columna).distinct().collect()]

# Cargar datos
with st.spinner("🔄 Inicializando Spark y cargando datos..."):
    df_spark = cargar_datos_spark()
    
    # Obtener valores para filtros
    sexos = ['Todos'] + obtener_valores_unicos(df_spark, 'SEXO')
    edad_min, edad_max = df_spark.select(F.min('EDAD'), F.max('EDAD')).collect()[0]

# Filtros interactivos
sexo_sel = st.sidebar.selectbox("⚥ Sexo", sexos)

rango_edad = st.sidebar.slider(
    "📊 Rango de edad",
    min_value=int(edad_min),
    max_value=int(edad_max),
    value=(int(edad_min), int(edad_max))
)

st.sidebar.markdown("---")
st.sidebar.subheader("💀 Resultado")
mostrar_fallecidos = st.sidebar.checkbox("Mostrar fallecidos", value=True)
mostrar_sobrevivientes = st.sidebar.checkbox("Mostrar sobrevivientes", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🫀 Comorbilidades")

comorbilidades = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 
                  'EPOC', 'INMUSUPR', 'RENAL_CRONICA', 'TABAQUISMO']

tipo_filtro = st.sidebar.radio(
    "Tipo de filtro",
    ["CON alguna", "SIN ninguna", "TODAS las seleccionadas"]
)

comorb_seleccionadas = []
if tipo_filtro == "TODAS las seleccionadas":
    for c in comorbilidades:
        if c in df_spark.columns and st.sidebar.checkbox(c):
            comorb_seleccionadas.append(c)
else:
    comorb_seleccionadas = comorbilidades

# ============================================
# APLICAR FILTROS
# ============================================
filtros = {
    'sexo': sexo_sel,
    'edad_min': rango_edad[0],
    'edad_max': rango_edad[1],
    'mostrar_fallecidos': mostrar_fallecidos,
    'mostrar_sobrevivientes': mostrar_sobrevivientes,
    'comorbilidades': comorb_seleccionadas,
    'tipo_filtro': tipo_filtro
}

with st.spinner("🔄 Aplicando filtros..."):
    df_filtrado = filtrar_datos_spark(df_spark, filtros)
    metricas = obtener_metricas(df_filtrado)

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================
st.header("📊 Resumen")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total pacientes", f"{metricas['total']:,}")

with col2:
    st.metric("Edad media", f"{metricas['edad_media']:.1f} años")

with col3:
    st.metric("Fallecidos", f"{metricas['fallecidos']:,}",
              delta=f"{metricas['fallecidos']/metricas['total']*100:.1f}%")

with col4:
    st.metric("Sobrevivientes", f"{metricas['sobrevivientes']:,}",
              delta=f"{metricas['sobrevivientes']/metricas['total']*100:.1f}%")

st.markdown("---")

# ============================================
# GRÁFICOS (convertir a pandas para visualización)
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Distribución por edad")
    # Tomar muestra para gráfico (10,000 filas es suficiente)
    muestra_pd = df_filtrado.select('EDAD', 'RESULTADO').sample(0.1).limit(10000).toPandas()
    
    fig_hist = px.histogram(
        muestra_pd,
        x='EDAD',
        nbins=50,
        color='RESULTADO',
        color_discrete_map={'Fallecido': '#e74c3c', 'Sobreviviente': '#2ecc71'},
        barmode='overlay'
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("🫀 Comorbilidades")
    # Contar comorbilidades
    comorb_counts = contar_comorbilidades(df_filtrado, comorbilidades)
    if comorb_counts:
        df_comorb = pd.DataFrame(comorb_counts, columns=['Comorbilidad', 'Conteo'])
        fig_comorb = px.bar(
            df_comorb,
            x='Comorbilidad',
            y='Conteo',
            color='Conteo',
            color_continuous_scale='Reds'
        )
        fig_comorb.update_layout(height=400)
        st.plotly_chart(fig_comorb, use_container_width=True)

# ============================================
# TABLA RESUMEN
# ============================================
with st.expander("📋 Ver muestra de datos"):
    # Mostrar solo 100 filas para no saturar
    muestra_df = df_filtrado.limit(100).toPandas()
    st.dataframe(muestra_df)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(f"⚡ Procesado con PySpark • {df_spark.count():,} registros totales")