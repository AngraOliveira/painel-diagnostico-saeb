import pandas as pd
import os

df = pd.read_csv(
    'data/diagnostico_habilidades_5EF.csv.gz',
    sep=';',
    compression='gzip',
)

print(df.columns)