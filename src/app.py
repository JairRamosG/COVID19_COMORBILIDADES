import streamlit as st
from pathlib import Path
import os
import pandas as pd

#st.set_page_config(
#    page_title="Dashboard COVID-19 Comorbilidades",
#    page_icon="",
#    layout="wide")


#st.title("🦠 Dashboard Interactivo de COVID-19 en México")
#st.markdown("""
#Este dashboard permite explorar los datos de pacientes COVID-19 en México, 
#analizando comorbilidades, distribución por edad, género y resultados.
#""")
#st.markdown("---")

#@st.cache_data
def cargar_datos():
    '''
    Cargar los datos que vienen en parquet
    '''

    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_PARQUET_PATH = Path(os.getenv('DATA_PARQUET', BASE_DIR / 'data' / 'parquet' / 'df_final.parquet'))

    if not DATA_PARQUET_PATH.exists():
        st.error(f'No se encuentra el archivo en la ruta: {DATA_PARQUET_PATH}')
        return None
    
    df = pd.read_parquet(DATA_PARQUET_PATH)
    return df

df = cargar_datos()