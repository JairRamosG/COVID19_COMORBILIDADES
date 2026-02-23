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


st.title("🦠 Dashboard Interactivo de COVID-19 en México")
st.markdown("""
Este dashboard permite explorar los datos de pacientes COVID-19 en México, 
analizando comorbilidades, distribución por edad, género y resultados.
""")
st.markdown("---")

@st.cache_resource
def cargar_datos():
    '''
    Carga el dataFrame de Spark de un parquet
    '''
    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_PARQUET_PATH = Path(os.getenv('DATA_PARQUET', BASE_DIR / 'data' / 'parquet' / 'df_final.parquet'))

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

# Filtro por edad
edad_min = df['EDAD'].min()
edad_max = df['EDAD'].max()
rango_edad = st.sidebar.slider('Rango de edades',
                               min_value = edad_min,
                               max_value = edad_max,
                               value = (edad_min, edad_max))

# Categoria de edad
categorias_edad = ['TODAS'] + sorted(df['CATEGORIAS_EDAD'].unique().tolist())
categorias_edad_sel = st.sidebar.selectbox('Categorias por edad', categorias_edad)