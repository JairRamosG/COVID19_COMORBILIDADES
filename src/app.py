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

@st.cache_resource
def cargar_datos():
    '''
    Carga el dataFrame de Spark de un parquet
    '''
    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_PARQUET_PATH = Path(os.getenv('DATA_PARQUET', BASE_DIR / 'data' / 'parquet'))

    if not DATA_PARQUET_PATH.exists():
        st.error(f'No se encuentra el archivo en la ruta: {DATA_PARQUET_PATH}')
        return None

    df = pd.read_parquet(str(DATA_PARQUET_PATH))
    
    return df

with st.spinner('Cargando datos'):
    df = cargar_datos()

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

comorbilidades = ['DIABETES', 'HIPERTENSION', 'OBESIDAD', 'ASMA', 'EPOC', 'INMUSUPR', 'RENAL_CRONICA', 'TABAQUISMO']
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

############################### MÉTRICAS PRINCIPALES #############################
st.header('Resumen General')
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Todal de pacientes",
              f"{len(df_filtrado):,}",
              delta = 'Registrados')

with col2:
    st.metric("Edad media",
              f"{df_filtrado['EDAD'].mean():.1f} años")

with col3:
    supervivientes = df_filtrado[df_filtrado['SOBREVIVIO'] == 1].shape[0]
    st.metric('Supervivencia',
              f"{supervivientes:,}")    
    
with col4:
    fallecimientos = df_filtrado[df_filtrado['SOBREVIVIO'] == 0].shape[0]
    st.metric('Fallecimientos',
              f"{fallecimientos:,}")    
    
with col5:
    comorb_prom = df['N_COMORBILIDADES'].mean()
    st.metric('Comorbilidades promedio',
              f"{comorb_prom:.1f}")

st.markdown('---')