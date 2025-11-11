import pandas as pd
import os 
import numpy as np
import re # Importação necessária para expressões regulares

# --- CONFIGURAÇÃO DE CAMINHOS ---
DIRETORIO_DADOS = 'D:/PI_SAEB/DADOS'

CAMINHO_ITENS = os.path.join(DIRETORIO_DADOS, 'TS_ITEM.csv') 

# Mapeamento de arquivos por série
ARQUIVOS_SERIES = {
    '5EF': {
        'respostas': os.path.join(DIRETORIO_DADOS, 'TS_ALUNO_5EF.csv'),
        'cluster': 'resultados_finais_5EF.csv', 
        'saida': 'diagnostico_habilidades_5EF.csv',
    },
    '9EF': {
        'respostas': os.path.join(DIRETORIO_DADOS, 'TS_ALUNO_9EF.csv'),
        'cluster': 'resultados_finais_9EF.csv', 
        'saida': 'diagnostico_habilidades_9EF.csv',
    }
}

# Constantes
COLUNA_ID_ALUNO = 'ID_ALUNO'
COLUNA_CLUSTER = 'CLUSTER'
COLUNA_DESCRITOR = 'NU_DESCRITOR_HABILIDADE'
COLUNA_DISCIPLINA = 'TP_DISCIPLINA'
COLUNA_GABARITO = 'TX_GABARITO'
COLUNA_POSICAO = 'NU_POSICAO' 
COLUNA_BLOCO = 'NU_BLOCO'
COLUNA_ID_ITEM = 'ID_ITEM' # Garante que a coluna esteja definida

# Tamanho dos blocos para evitar erro de memória (250 mil alunos por vez)
CHUNK_SIZE = 250000 

def criar_map_itens(df_itens):
    """Cria um mapeamento eficiente de bloco/posição para descritor/gabarito."""
    map_itens = {}
    for disc in ['LP', 'MT']:
        for bloco in [1, 2]:
            chave_coluna = f'TX_RESP_BLOCO{bloco}_{disc}'
            itens_do_bloco = df_itens[
                (df_itens[COLUNA_DISCIPLINA] == disc) & 
                (df_itens[COLUNA_BLOCO] == bloco)
            ].sort_values(by=COLUNA_POSICAO).reset_index(drop=True)
            
            for i, item_row in itens_do_bloco.iterrows():
                if item_row[COLUNA_GABARITO] not in ['X', 'E']:
                    map_itens[(chave_coluna, i)] = {
                        COLUNA_DESCRITOR: item_row[COLUNA_DESCRITOR],
                        COLUNA_GABARITO: item_row[COLUNA_GABARITO],
                        COLUNA_DISCIPLINA: disc
                    }
    return map_itens

def processar_chunk(df_chunk, map_itens):
    """Processa um bloco de alunos para gerar acertos/erros."""
    
    colunas_resp_processar = ['TX_RESP_BLOCO1_LP', 'TX_RESP_BLOCO2_LP', 
                              'TX_RESP_BLOCO1_MT', 'TX_RESP_BLOCO2_MT']
    
    df_chunk = df_chunk[[COLUNA_ID_ALUNO] + colunas_resp_processar].copy()

    def gerar_acertos_aluno(row):
        acertos_aluno = []
        id_aluno = row[COLUNA_ID_ALUNO]
        
        for col_resp in colunas_resp_processar:
            resp_str = str(row.get(col_resp, np.nan))
            
            if resp_str in ['nan', ''] or pd.isna(row.get(col_resp)):
                continue

            for i in range(len(resp_str)):
                chave = (col_resp, i)
                
                if chave in map_itens:
                    info = map_itens[chave]
                    resposta_aluno = resp_str[i]
                    gabarito = info[COLUNA_GABARITO]
                    
                    if resposta_aluno in ['A', 'B', 'C', 'D', 'E']:
                        acerto = 1 if resposta_aluno == gabarito else 0
                        
                        acertos_aluno.append({
                            COLUNA_ID_ALUNO: id_aluno,
                            COLUNA_DESCRITOR: info[COLUNA_DESCRITOR],
                            COLUNA_DISCIPLINA: info[COLUNA_DISCIPLINA],
                            'ACERTO': acerto
                        })
        return acertos_aluno

    resultados = df_chunk.apply(gerar_acertos_aluno, axis=1)
    
    # ACHATAMENTO DENTRO DO CHUNK: O gargalo de memória foi aqui!
    # A lista é achatada e convertida em DF DENTRO de cada chunk, liberando memória.
    lista_acertos = [item for sublist in resultados.tolist() for item in sublist]
    df_acertos_chunk = pd.DataFrame(lista_acertos)
    
    # Agrupar por Aluno, Descritor e Disciplina (para limpar e otimizar)
    if not df_acertos_chunk.empty:
        df_acertos_limpo = df_acertos_chunk.groupby([COLUNA_ID_ALUNO, COLUNA_DESCRITOR, COLUNA_DISCIPLINA])['ACERTO'].max().reset_index()
        return df_acertos_limpo
    return pd.DataFrame()


def gerar_diagnostico_habilidades_chunked(serie_config):
    """Função principal que gerencia o carregamento e processamento em blocos."""
    print(f"\n--- Iniciando diagnóstico {serie_config} com Chunking de {CHUNK_SIZE} ---")
    
    # Redefinir as constantes dentro do escopo da função para garantir que sejam reconhecidas
    COLUNA_ID_ITEM = 'ID_ITEM'
    COLUNA_GABARITO = 'TX_GABARITO'
    COLUNA_POSICAO = 'NU_POSICAO'
    COLUNA_BLOCO = 'NU_BLOCO'

    try:
        # 1. Carregar Metadados (TS_ITEM) e Clusters (fora do loop)
        df_itens = pd.read_csv(CAMINHO_ITENS, sep=';', encoding='latin-1') 
        df_itens = df_itens[[COLUNA_ID_ITEM, COLUNA_DESCRITOR, COLUNA_DISCIPLINA, 
                             COLUNA_GABARITO, COLUNA_POSICAO, COLUNA_BLOCO]].copy()
        df_itens.dropna(subset=[COLUNA_GABARITO], inplace=True)
        
        # 🛑 CORREÇÃO CRÍTICA: FILTRAR APENAS CÓDIGOS OFICIAIS SAEB (D<número>)
        # Isso garante que apenas os descritores que você tem a descrição oficial sejam processados.
        df_itens = df_itens[
            df_itens[COLUNA_DESCRITOR].astype(str).str.match(r'^D\d+$')
        ].copy()
        
        if df_itens.empty:
            print("AVISO: Após a filtragem, o arquivo TS_ITEM.csv não contém descritores no formato SAEB (D<número>). Verifique a matriz de referência usada.")
            return None
        
        df_clusters = pd.read_csv(ARQUIVOS_SERIES[serie_config]['cluster'], sep=';', encoding='latin-1')
        df_clusters = df_clusters[[COLUNA_ID_ALUNO, COLUNA_CLUSTER]].copy()
        df_clusters[COLUNA_ID_ALUNO] = df_clusters[COLUNA_ID_ALUNO].astype(str)
        
        map_itens = criar_map_itens(df_itens)

    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Verifique se {e.filename} existe e se os caminhos estão corretos.")
        return None

    # Lista para armazenar o diagnóstico de cada chunk
    diagnosticos_chunks = []
    
    # 2. Carregar Respostas em Blocos (Chunking)
    chunk_reader = pd.read_csv(ARQUIVOS_SERIES[serie_config]['respostas'], 
                               sep=';', encoding='latin-1', 
                               low_memory=False, iterator=True, chunksize=CHUNK_SIZE)
    
    chunk_num = 0
    for df_chunk_resp in chunk_reader:
        chunk_num += 1
        print(f"Processando Bloco {chunk_num}...")
        
        df_chunk_resp[COLUNA_ID_ALUNO] = df_chunk_resp[COLUNA_ID_ALUNO].astype(str)
        
        # Processar o chunk para gerar os acertos/erros
        df_acertos_chunk = processar_chunk(df_chunk_resp, map_itens)
        
        if not df_acertos_chunk.empty:
            # Unir com Clusters (somente o necessário)
            df_final_chunk = pd.merge(df_acertos_chunk, df_clusters, on=COLUNA_ID_ALUNO, how='inner')
            
            # Calcular a Média de Acerto (para o diagnóstico) para este chunk
            df_diagnostico_chunk = df_final_chunk.groupby([COLUNA_CLUSTER, COLUNA_DISCIPLINA, COLUNA_DESCRITOR])['ACERTO'].mean().reset_index()
            diagnosticos_chunks.append(df_diagnostico_chunk)
            
            # Limpar variáveis para liberar memória
            del df_chunk_resp, df_acertos_chunk, df_final_chunk, df_diagnostico_chunk
            
    # 3. Consolidar e Salvar
    if not diagnosticos_chunks:
        print("AVISO: Nenhum dado processado com sucesso. Verifique se o TS_ALUNO.csv tem respostas válidas.")
        return None
        
    df_consolidado = pd.concat(diagnosticos_chunks, ignore_index=True)
    
    # Calcular a Média Final de Acerto/Erro entre todos os chunks
    df_diagnostico_final = df_consolidado.groupby([COLUNA_CLUSTER, COLUNA_DISCIPLINA, COLUNA_DESCRITOR])['ACERTO'].mean().reset_index()
    
    df_diagnostico_final['TAXA_ERRO'] = 1 - df_diagnostico_final['ACERTO']
    df_diagnostico_final.drop(columns=['ACERTO'], inplace=True)
    
    # 4. Salvar Resultado
    df_diagnostico_final.to_csv(ARQUIVOS_SERIES[serie_config]['saida'], sep=';', encoding='latin-1', index=False)
    print(f"Diagnóstico de habilidades para {serie_config} concluído e salvo em '{ARQUIVOS_SERIES[serie_config]['saida']}'.")
    
    return df_diagnostico_final

if __name__ == "__main__":
    # Rodamos o 5EF
    df_5ef = gerar_diagnostico_habilidades_chunked('5EF')
    
    # Rodamos o 9EF
    df_9ef = gerar_diagnostico_habilidades_chunked('9EF')
    
    print("\nProcesso de Diagnóstico de Habilidades concluído para ambas as séries.")