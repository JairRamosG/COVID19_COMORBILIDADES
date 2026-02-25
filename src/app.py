import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import plotly.express as px 
import plotly.graph_objects as go


st.set_page_config(
    page_title="Dashboard COVID-19 Comorbilidades",
    page_icon="",
    layout="wide")


st.title("🦠 Dashboard de COVID-19 en México")
st.markdown("""
Este dashboard permite explorar los datos de pacientes COVID-19 en México analizando comorbilidades, distribución por edad, género y resultados.
""")
st.markdown("---")

@st.cache_data()
def cargar_datos():
    '''
    Carga el dataFrame de Spark de un parquet
    '''
    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_PARQUET = Path(os.getenv('DATA_PARQUET', BASE_DIR / 'data' / 'parquet'))

    if not DATA_PARQUET.exists():
        st.error(f'No se encuentra el archivo en la ruta: {DATA_PARQUET}')
        return None
    
    columnas_necesarias = ['EDAD', 'CATEGORIA_EDAD', 'SEXO', 'SOBREVIVIO', 
                          'N_COMORBILIDADES'] + comorbilidades

    df = pd.read_parquet(str(DATA_PARQUET), engine='pyarrow', columns=columnas_necesarias)

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

comorbilidades = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 'EPOC', 'INMUSUPR', 'RENAL_CRONICA', 'TABAQUISMO']

############################### CARGA DE DATOS #############################
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text('Cargando datos... Puede tomar algunos segundos')
df = cargar_datos()
progress_bar.progress(50)

status_text.text('Procesando datos')
progress_bar.progress(100)

status_text.empty()
progress_bar.empty()

if df is None:
    st.stop()

############################### SIDE BAR - FILTROS #############################
st.sidebar.header('Filtros')

# Filtro 1: Edad
edad_min = df['EDAD'].min()
edad_max = df['EDAD'].max()
filtro_edad = st.sidebar.slider('Rango de edades',
                               min_value = edad_min,
                               max_value = edad_max,
                               value = (edad_min, edad_max))

# Filtro 2: Categoria de edad
categorias_edad = ['TODAS'] + sorted(df['CATEGORIA_EDAD'].unique().tolist())
filtro_edad_categoria = st.sidebar.selectbox('Categorias por edad', categorias_edad)

# Filtro 3: Sexo
st.sidebar.markdown('---')
sexo_opciones = ['Todos','Masculino', 'Femenino']
filtro_sexo = st.sidebar.radio('Sexo', sexo_opciones)

# Filtro 4: Comorbilidades
st.sidebar.markdown('---')
st.sidebar.subheader('Comorbilidades')

filtro_comorbilidad = st.sidebar.radio('Tipo de filtro',
                                       ['Con alguna', 'Todas', 'Sin ninguna'],
                                       help = "Con alguna: al menos una\n" \
                                       "Todas: Tiene todas las comorbilidades\n" \
                                       "Sin ninguna: No tiene comorbilidades")
comorb_sel = st.sidebar.multiselect('Selecciona comorbilidades',
                                    options = comorbilidades,
                                    default = [])


############################### APLIICAR FILTROS #############################
df_filtrado = df.copy()

# Filtro por edad
df_filtrado = df_filtrado[
    (df_filtrado['EDAD'] >= filtro_edad[0]) &
    (df_filtrado['EDAD'] <= filtro_edad[1])
    ]

# Filtro por sexo
if filtro_sexo != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['SEXO'] == filtro_sexo]

############################### MÉTRICAS PRINCIPALES #############################
st.header('Resumen General')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Todal de pacientes",
              f"{len(df_filtrado):,}",
              delta = 'Registrados')

with col2:
    st.metric("Edad media",
              f"{df_filtrado['EDAD'].mean():.1f} años")

with col3:
    con_corm = df[(df['N_COMORBILIDADES'] >= 1)].shape[0]
    porcentaje_con_corm = (con_corm / len(df)) * 100
    st.metric('Padecen de alguna cormobilidad',
              f"{porcentaje_con_corm:.1f} %")


with col4:
    no_supervivientes = df_filtrado[df_filtrado['SOBREVIVIO'] == 0].shape[0]
    st.metric('Fallecimientos',
              f"{no_supervivientes:,}")    

st.markdown('---')

############################### GRÁFICOS #############################
st.subheader('Distribución por edades')

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        df_filtrado,
        x = 'EDAD',
        nbins = 50,
        color = 'SOBREVIVIO',
        title = 'Distribución por edades',
        color_discrete_map = {1 :  "#00ff6a",
                              0 : '#e74c3c'},
        barmode = 'overlay')
    fig_hist.update_layout(height = 400)
    st.plotly_chart(fig_hist)

with col2:
    fig_box = px.box(
        df_filtrado,
        x = 'SOBREVIVIO',
        y = 'EDAD',
        title = 'Edad por resultado',
        color = 'SOBREVIVIO',
        color_discrete_map = {1:  "#00ff6a",
                              0 : '#e74c3c'})
    fig_box.update_layout(height = 400)
    st.plotly_chart(fig_box)
