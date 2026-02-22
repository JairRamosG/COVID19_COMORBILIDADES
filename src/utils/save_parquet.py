from pathlib import Path
import os

def save_data(df, nombre):
    '''
    FUnción para guardar el dataset final en un archivo parquet
    Args:
        df     (DataFrame): DataFrame de PySpark a guardar
        nombre (str): nombre para el nuevo archivo parquet
    Returns:
        df_parquet (parquet): archivo formato parquet
    '''
    BASE_DIR = Path(os.getcwd()).parent
    PARQUET_DIR = BASE_DIR / 'data' / 'parquet' 

    PARQUET_DIR.parent.mkdir(parents=True, exist_ok=True)
    FILE_PATH = PARQUET_DIR / f'{nombre}.parquet'

    df.write.mode('overwrite').option('compression', 'snappy').parquet((str(FILE_PATH)))

    if FILE_PATH.exists():
        print(f'Archivo guardado en : {PARQUET_DIR}')
        print(f'Tamaño: {FILE_PATH.stat().st_size /  1e6:.f2} MB')
    else:
        print('No se pudo guardar el archivo')
    return str(FILE_PATH)