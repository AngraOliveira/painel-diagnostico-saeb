import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Constantes globais
SEED = 42
FEATURES_PRINCIPAIS = ['PROFICIENCIA_LP', 'PROFICIENCIA_MT']
COLUNAS_MANTER = [
    'ID_ALUNO', 
    'ID_ESCOLA', 
    'ID_UF', 
    'PROFICIENCIA_LP', 
    'PROFICIENCIA_MT',
    'TX_RESP_Q05a',   # Q05a: Deficiência
    'TX_RESP_Q05b',   # Q05b: TEA
    'TX_RESP_Q05c'    # Q05c: Superdotação
]

# --- DICIONÁRIO DE CONFIGURAÇÃO POR SÉRIE (MUDE OS VALORES DO 9EF!) ---
# Este dicionário precisa ser ajustado APÓS rodar e analisar os centroides do K-Means no 9EF.
# Os valores para o 5EF são baseados na sua análise original.
CONFIG_SERIES = {
    '5EF': {
        'N_CLUSTERS': 7, 
        # Mapeamento do 5EF (Baseado na tabela de centroides original)
        'ALTO_RISCO': ['1', '2', '3'], # Risco Extremo LP, Extremo Geral, Risco Moderado Equilibrado
        'MODERADO': ['5', '6'],        # Risco Discalculia, Risco Dislexia (MT >> LP)
        'NORMAL_BASE': ['4', '0']      # Alto Desempenho, Equilibrado/Geral
    },
    '9EF': {
        # **Ajuste este valor após o método do cotovelo/silhueta**
        'N_CLUSTERS': 7, 
        # **AJUSTE ESSES VALORES APÓS ANALISAR OS CENTROIDES REAIS DO 9EF**
        # Estes são PLACEHOLDERS e devem ser ajustados para a realidade do 9EF.
        'ALTO_RISCO': ['1', '2', '3'], 
        'MODERADO': ['5', '6'],
        'NORMAL_BASE': ['4', '0']
    }
}
# -----------------------------------------------------------------------


def classificar_risco_final(cluster_id_str, flag_risco_anomalia, tx_resp_q05c, config_risco):
    """
    Define o Status de Risco Final (Alto, Moderado, Normal, Superdotação)
    baseado no Cluster, na Anomalia e na Auto-declaração (Q05c), usando a 
    configuração de risco específica da série.
    """
    CLUSTERS_ALTO_RISCO = config_risco['ALTO_RISCO']
    CLUSTERS_MODERADO = config_risco['MODERADO']
    CLUSTERS_NORMAL_BASE = config_risco['NORMAL_BASE']

    # 1. Prioridade Máxima: SUPERDOTAÇÃO
    # Q05c == 'B' (Sim, tenho altas habilidades/Superdotação)
    if tx_resp_q05c == 'B' and cluster_id_str in CLUSTERS_NORMAL_BASE:
        return 'Superdotação'

    # 2. CLUSTERS DE ALTO RISCO
    if cluster_id_str in CLUSTERS_ALTO_RISCO:
        return 'Alto Risco'

    # 3. CLUSTERS DE RISCO MODERADO
    elif cluster_id_str in CLUSTERS_MODERADO:
        if flag_risco_anomalia == 'Risco':
            return 'Alto Risco' # Outlier em Moderado -> Alto
        else:
             return 'Risco Moderado'
    
    # 4. CLUSTER NORMAL
    elif cluster_id_str in CLUSTERS_NORMAL_BASE:
        if flag_risco_anomalia == 'Risco':
             return 'Risco Moderado' # Outlier em Normal -> Moderado
    
    # Caso de segurança
    else:
        return 'Normal'


def carregar_e_processar_dados(caminho_csv, ano_escolar, config_serie):
    """Carrega, limpa e processa os dados de proficiência para um dado ano."""
    print(f"Iniciando processamento para {ano_escolar} com K={config_serie['N_CLUSTERS']}...")
    
    try:
        # Tenta ler apenas as colunas necessárias para otimizar a memória
        df = pd.read_csv(caminho_csv, sep=';', encoding='latin-1', usecols=lambda x: x in COLUNAS_MANTER)
    except Exception as e:
        print(f"Erro ao carregar {caminho_csv}: {e}")
        return None

    # Limpeza, Seleção e Engenharia de Features
    df.dropna(subset=FEATURES_PRINCIPAIS, inplace=True)
    df.drop_duplicates(subset=['ID_ALUNO'], keep='first', inplace=True)
    
    # Preenchimento de 'TX_RESP_Q05c' (Não-Superdotação)
    df['TX_RESP_Q05c'] = df['TX_RESP_Q05c'].fillna('B')
    
    # Engenharia de Features
    df['DISCREPANCIA'] = df['PROFICIENCIA_LP'] - df['PROFICIENCIA_MT']
    
    # Normalização para o Clustering e Isolation Forest
    features_modelo = FEATURES_PRINCIPAIS + ['DISCREPANCIA'] 
    df_modelo = df[features_modelo].copy()
    
    scaler = StandardScaler()
    df_modelo_scaled = scaler.fit_transform(df_modelo)
    
    # 4. Treinamento do K-Means (Clustering)
    # LINHA CORRIGIDA: Removido o artefato
    kmeans = KMeans(n_clusters=config_serie['N_CLUSTERS'], random_state=SEED, n_init=10)
    df['CLUSTER'] = kmeans.fit_predict(df_modelo_scaled)
    df['CLUSTER'] = df['CLUSTER'].astype(str)
    
    # 5. Treinamento do Isolation Forest (Detecção de Risco/Anomalia)
    iso_forest = IsolationForest(contamination=0.05, random_state=SEED)
    df['ANOMALIA'] = iso_forest.fit_predict(df_modelo_scaled)
    df['FLAG_RISCO_ANOMALIA'] = df['ANOMALIA'].apply(lambda x: 'Normal' if x == 1 else 'Risco')

    # Geração do Status de Risco customizado (Alto, Moderado, Normal, Superdotação)
    # Passando a configuração da série como novo argumento
    df['STATUS_RISCO_FINAL'] = df.apply(
        lambda row: classificar_risco_final(
            row['CLUSTER'], 
            row['FLAG_RISCO_ANOMALIA'],
            row['TX_RESP_Q05c'],
            config_serie # Novo argumento de configuração
        ), 
        axis=1
    )

    print(f"Processamento para {ano_escolar} concluído. Total de alunos: {len(df)}")
    return df

# --- Execução Principal ---
if __name__ == "__main__":
    CAMINHO_5EF = 'D:/PI_SAEB/DADOS/TS_ALUNO_5EF.csv'
    CAMINHO_9EF = 'D:/PI_SAEB/DADOS/TS_ALUNO_9EF.csv'
    
    # Processa 5EF usando a sua configuração de 5EF
    df_5ef_analisado = carregar_e_processar_dados(CAMINHO_5EF, '5º Ano', CONFIG_SERIES['5EF'])
    
    # Processa 9EF usando a configuração de 9EF (precisa ser ajustada/analisada antes)
    df_9ef_analisado = carregar_e_processar_dados(CAMINHO_9EF, '9º Ano', CONFIG_SERIES['9EF'])
    
    if df_5ef_analisado is not None:
        df_5ef_analisado.to_csv('resultados_finais_5EF.csv', sep=';', encoding='latin-1', index=False)
        print("Arquivo 'resultados_finais_5EF.csv' salvo com sucesso.")
         
    if df_9ef_analisado is not None:
        df_9ef_analisado.to_csv('resultados_finais_9EF.csv', sep=';', encoding='latin-1', index=False)
        print("Arquivo 'resultados_finais_9EF.csv' salvo com sucesso.")

    print("\nProcesso de Análise concluído.")