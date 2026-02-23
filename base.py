# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Dashboard COVID-19 México",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦠 Dashboard COVID-19 México - Análisis de Comorbilidades")
st.markdown("---")

# ============================================
# CARGA DE DATOS (CON CACHÉ DE STREAMLIT)
# ============================================
@st.cache_data
def cargar_datos():
    """Carga el DataFrame desde Parquet usando pandas"""
    # Ruta relativa desde src/ a data/
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "parquet" / "df_final.parquet"
    
    if not DATA_PATH.exists():
        st.error(f"❌ No se encontró el archivo de datos en: {DATA_PATH}")
        st.info("""
        Para generar los datos:
        1. Ejecuta el procesamiento con PySpark en local
        2. Guarda el resultado en data/parquet/df_final.parquet
        3. Vuelve a desplegar
        """)
        return None
    
    # Cargar con pandas (optimizado para Parquet)
    df = pd.read_parquet(DATA_PATH)
    
    # Crear columnas auxiliares para visualización
    if 'SOBREVIVIO' in df.columns:
        df['RESULTADO'] = df['SOBREVIVIO'].map({1: '✅ Sobrevivió', 0: '❌ Falleció'})
    
    if 'SEXO' in df.columns:
        df['SEXO_LABEL'] = df['SEXO'].map({1: 'Mujer', 2: 'Hombre'})
    
    return df

# ============================================
# CARGA DE DATOS
# ============================================
with st.spinner("🔄 Cargando datos..."):
    df = cargar_datos()

if df is None:
    st.stop()

# ============================================
# SIDEBAR - FILTROS INTERACTIVOS
# ============================================
st.sidebar.header("🔍 Filtros de Análisis")

# Filtro 1: Rango de edad
edad_min = int(df['EDAD'].min())
edad_max = int(df['EDAD'].max())
rango_edad = st.sidebar.slider(
    "📊 Rango de edad",
    min_value=edad_min,
    max_value=edad_max,
    value=(edad_min, edad_max)
)

# Filtro 2: Categoría de edad
categorias_edad = ['Todas'] + sorted(df['CATEGORIA_EDAD'].unique().tolist())
categoria_sel = st.sidebar.selectbox("👥 Categoría de edad", categorias_edad)

# Filtro 3: Sexo
sexo_opciones = ['Todos', 'Mujer', 'Hombre']
sexo_sel = st.sidebar.radio("⚥ Sexo", sexo_opciones, horizontal=True)

# Filtro 4: Resultado
resultado_opciones = ['Todos', 'Sobrevivió', 'Falleció']
resultado_sel = st.sidebar.radio("💀 Resultado", resultado_opciones, horizontal=True)

# Filtro 5: Nivel de riesgo
st.sidebar.markdown("---")
st.sidebar.subheader("⚕️ Nivel de Riesgo")
riesgo_opciones = ['Todos'] + sorted(df['RIESGO'].unique().tolist())
riesgo_sel = st.sidebar.multiselect(
    "Selecciona niveles de riesgo",
    options=riesgo_opciones,
    default=['Todos']
)

# Filtro 6: Comorbilidades
st.sidebar.markdown("---")
st.sidebar.subheader("🫀 Comorbilidades")

comorbilidades = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 
                  'EPOC', 'INMUSUPR', 'RENAL_CRONICA', 'TABAQUISMO']

tipo_filtro_comorb = st.sidebar.radio(
    "Tipo de filtro",
    ["CON alguna", "TODAS", "SIN ninguna"],
    help="CON alguna: pacientes con al menos una\nTODAS: pacientes con todas\nSIN ninguna: pacientes sin ninguna"
)

comorb_sel = st.sidebar.multiselect(
    "Selecciona comorbilidades",
    options=comorbilidades,
    default=[]
)

# ============================================
# APLICAR FILTROS
# ============================================
df_filtrado = df.copy()

# Filtro por edad
df_filtrado = df_filtrado[
    (df_filtrado['EDAD'] >= rango_edad[0]) & 
    (df_filtrado['EDAD'] <= rango_edad[1])
]

# Filtro por categoría de edad
if categoria_sel != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['CATEGORIA_EDAD'] == categoria_sel]

# Filtro por sexo
if sexo_sel != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['SEXO_LABEL'] == sexo_sel]

# Filtro por resultado
if resultado_sel != 'Todos':
    resultado_map = {'Sobrevivió': 1, 'Falleció': 0}
    df_filtrado = df_filtrado[df_filtrado['SOBREVIVIO'] == resultado_map[resultado_sel]]

# Filtro por riesgo
if 'Todos' not in riesgo_sel:
    df_filtrado = df_filtrado[df_filtrado['RIESGO'].isin(riesgo_sel)]

# Filtro por comorbilidades
if comorb_sel:
    if tipo_filtro_comorb == "CON alguna":
        # OR - al menos una
        mask = pd.Series(False, index=df_filtrado.index)
        for c in comorb_sel:
            mask |= (df_filtrado[c] == 1)
        df_filtrado = df_filtrado[mask]
    
    elif tipo_filtro_comorb == "TODAS":
        # AND - todas las seleccionadas
        for c in comorb_sel:
            df_filtrado = df_filtrado[df_filtrado[c] == 1]
    
    else:  # "SIN ninguna"
        # NOT - ninguna de las seleccionadas
        mask = pd.Series(True, index=df_filtrado.index)
        for c in comorb_sel:
            mask &= (df_filtrado[c] != 1)
        df_filtrado = df_filtrado[mask]

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================
st.header("📊 Resumen General")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total pacientes",
        f"{len(df_filtrado):,}",
        delta=f"{len(df_filtrado)/len(df)*100:.1f}% del total"
    )

with col2:
    st.metric(
        "Edad media",
        f"{df_filtrado['EDAD'].mean():.1f} años"
    )

with col3:
    sobrev = df_filtrado[df_filtrado['SOBREVIVIO'] == 1].shape[0]
    st.metric(
        "✅ Sobrevivieron",
        f"{sobrev:,}",
        delta=f"{sobrev/len(df_filtrado)*100:.1f}%" if len(df_filtrado) > 0 else "0%"
    )

with col4:
    fallecidos = df_filtrado[df_filtrado['SOBREVIVIO'] == 0].shape[0]
    st.metric(
        "❌ Fallecidos",
        f"{fallecidos:,}",
        delta=f"{fallecidos/len(df_filtrado)*100:.1f}%" if len(df_filtrado) > 0 else "0%"
    )

with col5:
    st.metric(
        "Comorbilidades promedio",
        f"{df_filtrado['N_COMORBILIDADES'].mean():.2f}"
    )

st.markdown("---")

# ============================================
# GRÁFICO 1: Distribución por edad
# ============================================
st.subheader("📈 Distribución por Edad y Resultado")

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        df_filtrado,
        x='EDAD',
        nbins=50,
        color='RESULTADO',
        title="Distribución de edades",
        color_discrete_map={'✅ Sobrevivió': '#2ecc71', '❌ Falleció': '#e74c3c'},
        barmode='overlay'
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(
        df_filtrado,
        x='RESULTADO',
        y='EDAD',
        title="Edad por resultado",
        color='RESULTADO',
        color_discrete_map={'✅ Sobrevivió': '#2ecc71', '❌ Falleció': '#e74c3c'}
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
            count = df_filtrado[df_filtrado[c] == 1].shape[0]
            pct = (count / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
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
    comorb_dist = df_filtrado['N_COMORBILIDADES'].value_counts().sort_index().reset_index()
    comorb_dist.columns = ['N_COMORBILIDADES', 'Conteo']
    
    fig_dist = px.bar(
        comorb_dist,
        x='N_COMORBILIDADES',
        y='Conteo',
        title="Distribución del número de comorbilidades",
        color='Conteo',
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
tabla_mortalidad = df_filtrado.groupby('CATEGORIA_EDAD').agg(
    total=('EDAD', 'count'),
    sobrevivieron=('SOBREVIVIO', lambda x: (x == 1).sum()),
    fallecidos=('SOBREVIVIO', lambda x: (x == 0).sum())
).reset_index()

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
        # Datos para stacked bar
        df_stack = tabla_mortalidad.melt(
            id_vars=['CATEGORIA_EDAD'], 
            value_vars=['sobrevivieron', 'fallecidos'],
            var_name='Resultado',
            value_name='Cantidad'
        )
        df_stack['Resultado'] = df_stack['Resultado'].map({
            'sobrevivieron': '✅ Sobrevivió',
            'fallecidos': '❌ Falleció'
        })
        
        fig_stacked = px.bar(
            df_stack,
            x='CATEGORIA_EDAD',
            y='Cantidad',
            color='Resultado',
            title="Sobrevivientes vs Fallecidos por categoría de edad",
            color_discrete_map={'✅ Sobrevivió': '#2ecc71', '❌ Falleció': '#e74c3c'},
            barmode='stack'
        )
        fig_stacked.update_layout(height=400)
        st.plotly_chart(fig_stacked, use_container_width=True)

# ============================================
# GRÁFICO 4: Análisis de Riesgo
# ============================================
st.subheader("⚕️ Análisis de Nivel de Riesgo")

riesgo_analysis = df_filtrado.groupby('RIESGO').agg(
    total=('EDAD', 'count'),
    edad_promedio=('EDAD', 'mean'),
    sobrevivieron=('SOBREVIVIO', lambda x: (x == 1).sum()),
    fallecidos=('SOBREVIVIO', lambda x: (x == 0).sum())
).reset_index()

if not riesgo_analysis.empty:
    riesgo_analysis['tasa_mortalidad'] = (riesgo_analysis['fallecidos'] / riesgo_analysis['total'] * 100).round(1)
    
    # Ordenar por nivel de riesgo
    orden_riesgo = {'NORMAL': 1, 'MODERADO': 2, 'ALTO': 3, 'EXTREMO': 4}
    riesgo_analysis['orden'] = riesgo_analysis['RIESGO'].map(orden_riesgo)
    riesgo_analysis = riesgo_analysis.sort_values('orden').drop('orden', axis=1)
    
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
    # Seleccionar columnas más relevantes
    columnas_mostrar = ['EDAD', 'CATEGORIA_EDAD', 'SEXO_LABEL', 'RIESGO', 'RESULTADO',
                        'N_COMORBILIDADES', 'DIABETES', 'HIPERTENSION', 'OBESIDAD']
    
    columnas_existentes = [c for c in columnas_mostrar if c in df_filtrado.columns]
    
    st.dataframe(
        df_filtrado[columnas_existentes].head(1000),
        use_container_width=True,
        height=400
    )
    st.caption(f"Mostrando primeras 1000 filas de {len(df_filtrado):,} totales")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📊 Total en base: {len(df):,} registros")
with col2:
    st.caption(f"🔍 Mostrando: {len(df_filtrado):,} registros")
with col3:
    st.caption("🦠 Datos COVID-19 México - Análisis de Comorbilidades")
    st.caption(f"📅 Última actualización: {pd.Timestamp.now().strftime('%d/%m/%Y')}")