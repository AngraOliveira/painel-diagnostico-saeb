import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
import os
import numpy as np
import re 
from typing import Dict, Any, Tuple

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel de Diagnóstico SAEB",
    page_icon="📊",
    layout="wide"
)

# --- Constantes Globais (Mantidas) ---
ARQUIVOS_SERIES = {
    '5EF': {
        'resultados': 'data/resultados_finais_5EF.csv.gz',
        'diagnostico': 'data/diagnostico_habilidades_5EF.csv.gz',
        'matriz': 'descritores_5EF.csv'  
    },
    '9EF': {
        'resultados': 'data/resultados_finais_9EF.csv.gz',
        'diagnostico': 'data/diagnostico_habilidades_9EF.csv.gz',
        'matriz': 'descritores_9EF.csv' 
    }
}
HABILIDADES_OCULTAR = {
    '5EF': {
        'LP': ['D20', 'D18'], 
        'MT': ['D34', 'D35', 'D36'] 
    },
    '9EF': {
        'LP': [],
        'MT': []
    }
}
MAPA_UF = {
    11: 'RO - Rondônia', 12: 'AC - Acre', 13: 'AM - Amazonas', 14: 'RR - Roraima', 
    15: 'PA - Pará', 16: 'AP - Amapá', 17: 'TO - Tocantins', 21: 'MA - Maranhão', 
    22: 'PI - Piauí', 23: 'CE - Ceará', 24: 'RN - Rio Grande do Norte', 25: 'PB - Paraíba', 
    26: 'PE - Pernambuco', 27: 'AL - Alagoas', 28: 'SE - Sergipe', 29: 'BA - Bahia', 
    31: 'MG - Minas Gerais', 32: 'ES - Espírito Santo', 33: 'RJ - Rio de Janeiro', 
    35: 'SP - São Paulo', 41: 'PR - Paraná', 42: 'SC - Santa Catarina', 
    43: 'RS - Rio Grande do Sul', 50: 'MS - Mato Grosso do Sul', 51: 'MT - Mato Grosso', 
    52: 'GO - Goiás', 53: 'DF - Distrito Federal', 
    -1: 'Não Informado' 
}
CONFIG_APP_SERIES = { # Dicionário de configuração por série
    '5EF': {
        'CLUSTER_LEGEND': {
            '3': 'Dificuldade Crítica Generalizada', '1': 'Risco Extremo em LP (Dislexia)',   
            '2': 'Risco Extremo em MT (Discalculia)', '0': 'Abaixo da Média Equilibrado',      
            '6': 'Risco Extremo LP (Dislexia Forte)', '5': 'Risco MT (Discalculia Moderado)',   
            '4': 'Alto Desempenho Equilibrado',       
        },
        'RISCO_PARA_CLUSTER': {
            'Alto Risco': ['3', '1', '2', '6'], 'Risco Moderado': ['0', '5'], 'Normal': ['4']          
        },
        'CLUSTER_PARA_RISCO': {
            '3': 'Alto Risco', '1': 'Alto Risco', '2': 'Alto Risco', '6': 'Alto Risco',
            '0': 'Risco Moderado', '5': 'Risco Moderado', '4': 'Normal'
        }
    },
    '9EF': {
        'CLUSTER_LEGEND': {
            '0': 'Grande Déficit em MT (Crítico)', '6': 'Grande Déficit em LP (Crítico)',
            '3': 'Risco Extremo Dislexia (LP << MT)', '2': 'Risco Extremo Discalculia (LP >> MT)',
            '5': 'Alto Desempenho Discrepante (LP Forte)', '1': 'Alto Desempenho Discrepante (MT Forte)',
            '4': 'Perfil Mediano Equilibrado (Média Geral)', 
        },
        'RISCO_PARA_CLUSTER': {
            'Alto Risco': ['0', '6', '3', '2'], 'Risco Moderado': ['5', '1'], 'Normal': ['4']         
        },
        'CLUSTER_PARA_RISCO': {
            '4': 'Normal', '0': 'Alto Risco', '6': 'Alto Risco',
            '3': 'Alto Risco', '2': 'Alto Risco', '5': 'Risco Moderado', 
            '1': 'Risco Moderado'
        }
    }
}
STATUS_RISCO_FINAL = ['Normal', 'Risco Moderado', 'Alto Risco', 'Superdotação']
RISK_SORT_KEY = {'Alto Risco': 0, 'Risco Moderado': 1, 'Superdotação': 2, 'Normal': 3, 'Desconhecido': 99}
COR_PRIMARIA_AZUL = '#1f77b4' 
COR_SECUNDARIA_VERDE = '#2ca02c' 
COR_ALTO_RISCO = '#17becf' 
COR_MODERADO = '#9467bd' 
COR_NORMAL = COR_SECUNDARIA_VERDE 
COR_SUPERDOTACAO = '#ff7f0e' 

# --- CSS Customizado (Mantido) ---
st.markdown("""
<style>
    /* ... CSS Mantido ... */
    .stMultiSelect div[data-baseweb="select"] > div:first-child > div > div[data-baseweb="tag"] {
        background-color: #1f77b4 !important; 
        color: white !important; 
    }
    /* ... demais estilos CSS ... */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: #1f77b4 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:focus:not(:active) {
        border-bottom-color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Funções Auxiliares (Mantidas) ---

def set_clusters_from_risco(config_serie: Dict[str, Any]):
    """Atualiza a seleção de clusters baseada na seleção de status de risco."""
    riscos_selecionados = st.session_state.filtro_status_risco_global_temp
    clusters_a_selecionar = set()
    risco_para_cluster = config_serie['RISCO_PARA_CLUSTER']
 
    for risco in riscos_selecionados:
        clusters_a_selecionar.update(risco_para_cluster.get(risco, []))
    st.session_state.filtro_cluster_global_temp = list(clusters_a_selecionar)

def set_risco_from_clusters(todos_status: list, config_serie: Dict[str, Any]):
    """Atualiza a seleção de status de risco baseada na seleção de clusters."""
    clusters_selecionados = st.session_state.filtro_cluster_global_temp
    riscos_a_selecionar = set()
    cluster_para_risco = config_serie['CLUSTER_PARA_RISCO']

    for cluster in clusters_selecionados:
        riscos_a_selecionar.add(cluster_para_risco.get(cluster))

    if not riscos_a_selecionar:
        st.session_state.filtro_status_risco_global_temp = todos_status
    else:
        riscos_ordenados = [r for r in todos_status if r in riscos_a_selecionar]
        st.session_state.filtro_status_risco_global_temp = riscos_ordenados

def limpar_caracteres_acentuados(texto: Any) -> Any:
    """Função para tentar corrigir problemas de codificação de caracteres."""
    if pd.isna(texto) or not isinstance(texto, str):
        return texto
    
    # As correções do seu código original
    texto = texto.replace('Ã£o', 'ão').replace('Ã£o', 'ão')
    texto = texto.replace('Ãªncia', 'ência')
    texto = texto.replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã­', 'í').replace('Ã³', 'ó').replace('Ãº', 'ú')
    texto = texto.replace('Ã§', 'ç').replace('Ãµ', 'õ')
    texto = texto.replace('Ão', 'ão').replace('Ãa', 'ã') 
    # Tentativa de correção genérica para padrões comuns de latin-1 corrompido em utf-8
    texto = re.sub(r'Ã\w+', lambda m: m.group(0).lstrip('Ã'), texto) 
    return texto

@st.cache_data
def carregar_dados(serie: str) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Função para carregar todos os dataframes necessários."""
    
    COLUNA_DESCRITOR_MATRIZ = 'NU_DESCRITOR_HABILIDADE' 
    COLUNA_DESCRICAO_MATRIZ = 'DESCRICAO'
    COLUNA_DISCIPLINA_MATRIZ = 'TP_DISCIPLINA' 
    SEPARADOR_CSV = ';'
    caminho_resultados = ARQUIVOS_SERIES[serie]['resultados']
    caminho_diagnostico = ARQUIVOS_SERIES[serie]['diagnostico']
    caminho_matriz = ARQUIVOS_SERIES[serie]['matriz']

    if not os.path.exists(caminho_resultados) or not os.path.exists(caminho_diagnostico):
        st.error(f"ERRO: Arquivos principais da série **{serie}** não encontrados. Verifique se eles estão em 'data/'")
        return None, None

    try:
        df_alunos = pd.read_csv(
            caminho_resultados,
            sep=SEPARADOR_CSV,
            encoding='latin-1',
            compression='gzip', 
            dtype={'ID_ALUNO': str, 'CLUSTER': str}
        )
    except Exception as e:
        st.error(f"Erro ao ler **{caminho_resultados}**. Detalhe: {e}")
        return None, None
        
    df_alunos['STATUS_RISCO_FINAL'] = df_alunos['STATUS_RISCO_FINAL'].astype(str).replace('nan', 'Normal') 
    
    df_alunos['ID_UF'] = pd.to_numeric(df_alunos['ID_UF'], errors='coerce').fillna(-1).astype(int)
    df_alunos['UF_DESCRICAO'] = df_alunos['ID_UF'].map(MAPA_UF).fillna('UF Desconhecida')

    try:
        df_diagnostico = pd.read_csv(
            caminho_diagnostico,
            sep=SEPARADOR_CSV,
            encoding='latin-1',
            compression='gzip', 
            dtype={'CLUSTER': str, COLUNA_DESCRITOR_MATRIZ: str}
        )
        
        df_matriz = pd.read_csv(
            caminho_matriz,
            sep=SEPARADOR_CSV,
            encoding='latin-1'
        )
    except Exception as e:
        st.error(f"Erro ao carregar arquivos de diagnóstico/matriz. Detalhe: {e}")
        return None, None

    try:
        COLUNAS_MESCLAGEM = [COLUNA_DESCRITOR_MATRIZ, COLUNA_DISCIPLINA_MATRIZ]

        df_diag_completo = pd.merge(
            df_diagnostico,
            df_matriz,
            left_on=COLUNAS_MESCLAGEM,
            right_on=COLUNAS_MESCLAGEM, 
            how='left'
        )
    except KeyError as e:
        st.error(f"KeyError durante o Merge: A coluna **{e}** não foi encontrada no seu arquivo de descritores.")
        return None, None
    
    COLUNA_DISCIPLINA_MATRIZ = 'TP_DISCIPLINA' # Definida no topo da função
    df_diag_completo[COLUNA_DISCIPLINA_MATRIZ] = df_diag_completo[COLUNA_DISCIPLINA_MATRIZ].astype(str).str.strip()

    # Garante o nome padrão para a coluna de descrição da habilidade
    if COLUNA_DESCRICAO_MATRIZ != 'DESCRICAO_HABILIDADE' and COLUNA_DESCRICAO_MATRIZ in df_diag_completo.columns:
        df_diag_completo.rename(
            columns={COLUNA_DESCRICAO_MATRIZ: 'DESCRICAO_HABILIDADE'}, 
            inplace=True
        )
    
    df_diag_completo['DESCRICAO_HABILIDADE'] = df_diag_completo['DESCRICAO_HABILIDADE'].apply(limpar_caracteres_acentuados)

    df_diag_completo['DESCRICAO_HABILIDADE'] = df_diag_completo['DESCRICAO_HABILIDADE'].fillna(
        df_diag_completo[COLUNA_DESCRITOR_MATRIZ] + " (Descrição não disponível)"
    )

    return df_alunos, df_diag_completo

# --- Funções de Visualização (NOVO) ---

def criar_kpis_visao_geral(df_alunos_filtrado: pd.DataFrame):
    """Exibe os KPIs principais na Visão Geral."""
    st.subheader("Métricas Principais")
    kpi_t1, kpi_t2, kpi_t3, kpi_t4 = st.columns(4) 
    
    total_alunos = df_alunos_filtrado['ID_ALUNO'].nunique()
    alunos_risco = df_alunos_filtrado[
        df_alunos_filtrado['STATUS_RISCO_FINAL'].isin(['Alto Risco', 'Risco Moderado'])
    ]['ID_ALUNO'].nunique()
    
    proficiencia_lp = df_alunos_filtrado['PROFICIENCIA_LP'].mean()
    proficiencia_mt = df_alunos_filtrado['PROFICIENCIA_MT'].mean()
    
    kpi_t1.metric("Total de Alunos", f"{total_alunos:,}".replace(",", "."))
    kpi_t2.metric("Alunos em Risco (Alto ou Mod.)", f"{alunos_risco:,}".replace(",", "."))
    kpi_t3.metric("Proficiência Média (LP)", f"{proficiencia_lp:.2f}")
    kpi_t4.metric("Proficiência Média (MT)", f"{proficiencia_mt:.2f}") 
    
    st.divider()

def criar_grafico_risco(df_alunos_filtrado: pd.DataFrame):
    """Gera e exibe o gráfico de pizza de Status de Risco."""
    st.markdown("#### Distribuição por Status de Risco")
    df_pizza = df_alunos_filtrado['STATUS_RISCO_FINAL'].value_counts().reset_index()
    
    color_map_risco = {
        'Alto Risco': COR_ALTO_RISCO, 'Risco Moderado': COR_MODERADO, 
        'Normal': COR_NORMAL, 'Superdotação': COR_SUPERDOTACAO
    }
    
    fig_pizza = px.pie(
        df_pizza,
        names='STATUS_RISCO_FINAL',
        values='count',
        title="Alunos por Status de Risco",
        hole=0.3,
        color='STATUS_RISCO_FINAL',
        color_discrete_map=color_map_risco
    )
    st.plotly_chart(fig_pizza, use_container_width=True)

def criar_grafico_cluster(df_alunos_filtrado: pd.DataFrame, cluster_legend: Dict[str, str]):
    """Gera e exibe o gráfico de barras de Contagem por Cluster."""
    st.markdown("#### Contagem por Cluster")
    df_barras = df_alunos_filtrado['CLUSTER'].value_counts().reset_index().sort_values(by='CLUSTER')
    
    df_barras['CLUSTER_DESCRICAO'] = df_barras['CLUSTER'].map(cluster_legend).fillna(
        df_barras['CLUSTER'].apply(lambda c: f"Cluster {c} (Sem Legenda)")
    )
    
    fig_barras = px.bar(
        df_barras,
        x='CLUSTER_DESCRICAO', 
        y='count',
        title="Alunos por Cluster",
        labels={'count': 'Número de Alunos', 'CLUSTER_DESCRICAO': 'Cluster de Risco'},
        color_discrete_sequence=[COR_PRIMARIA_AZUL] 
    )
    fig_barras.update_xaxes(tickangle=45) 
    st.plotly_chart(fig_barras, use_container_width=True)

def exibir_legenda_clusters(cluster_legend: Dict[str, str], cluster_para_risco: Dict[str, str]):
    """Exibe a legenda ordenada dos perfis de cluster."""
    with st.expander("ℹ️ Legenda dos Perfis de Risco", expanded=False):
        lista_legendas = []
        for cluster_id, descricao in cluster_legend.items():
            status_risco = cluster_para_risco.get(cluster_id, 'Desconhecido')
            
            lista_legendas.append((
                RISK_SORT_KEY.get(status_risco, 99), # Chave de ordenação
                cluster_id, 
                descricao, 
                status_risco
            ))
        
        lista_legendas.sort(key=lambda x: x[0]) 
        
        for _, cluster_id, descricao, status_risco in lista_legendas:
            st.markdown(f"**[{cluster_id}] {status_risco}**: {descricao}")

def criar_grafico_dispersao(df_alunos_filtrado: pd.DataFrame, cluster_legend: Dict[str, str]):
    """Gera e exibe o gráfico de dispersão LP vs MT."""
    st.markdown("#### Proficiência (LP vs MT) por Cluster")
    
    df_alunos_filtrado['DESCRICAO_CLUSTER'] = df_alunos_filtrado['CLUSTER'].map(cluster_legend).fillna('Desconhecido')
    
    # Amostra de dados (mantido para performance)
    df_scatter = df_alunos_filtrado.sample(min(len(df_alunos_filtrado), 5000), random_state=42)
    
    fig_scatter = px.scatter(
        df_scatter,
        x='PROFICIENCIA_LP',
        y='PROFICIENCIA_MT',
        color='DESCRICAO_CLUSTER', 
        hover_data=['ID_ALUNO', 'DESCRICAO_CLUSTER', 'STATUS_RISCO_FINAL', 'UF_DESCRICAO'],
        title="Relação entre Proficiência LP e MT por Perfil (Cluster)",
        color_discrete_sequence=px.colors.qualitative.Plotly 
    )
    
    fig_scatter.update_layout(
        legend_title_text='Cluster de Risco',
        xaxis_title='Proficiência Língua Portuguesa (LP)',
        yaxis_title='Proficiência Matemática (MT)',
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

def criar_heatmap_habilidade(df_diag_filtrado: pd.DataFrame, cluster_legend: Dict[str, str], disciplina_selec: str):
    """Gera e exibe o mapa de calor da Taxa de Erro por Habilidade e Cluster."""
    st.subheader("Mapa de Calor: Taxa de Erro por Habilidade e Cluster")
    st.markdown("O Eixo Y mostra o **código da Habilidade**. Quanto mais escuro (verde intenso), maior a taxa de erro média do cluster naquela habilidade. **Passe o mouse na célula para ver o código e a descrição completa.**")
    
    try:
        df_heatmap_pivot = df_diag_filtrado.pivot_table(
            index='NU_DESCRITOR_HABILIDADE', 
            columns='CLUSTER',
            values='TAXA_ERRO'
        )
        
        z_values = df_heatmap_pivot.values
        x_clusters = df_heatmap_pivot.columns.tolist()
        y_descritores = df_heatmap_pivot.index.tolist()
        
        df_descricoes = df_diag_filtrado[['NU_DESCRITOR_HABILIDADE', 'DESCRICAO_HABILIDADE']].drop_duplicates()
        descricoes_lookup = df_descricoes.set_index('NU_DESCRITOR_HABILIDADE')['DESCRICAO_HABILIDADE'].to_dict()
        
        hover_text = []
        annotations = []

        for i, descritor in enumerate(y_descritores):
            row_text = []
            descricao = limpar_caracteres_acentuados(descricoes_lookup.get(descritor, "Descrição não encontrada"))
            
            for j, cluster in enumerate(x_clusters):
                erro = df_heatmap_pivot.loc[descritor, cluster]
                cluster_desc = cluster_legend.get(cluster, f"Cluster {cluster}") 
                
                if not pd.isna(erro):
                    hover_item = (
                        f"<b>Cluster:</b> {cluster} ({cluster_desc})<br>"
                        f"<b>Código:</b> {descritor}<br>"
                        f"<b>Habilidade:</b> {descricao}<br>"
                        f"<b>Taxa de Erro:</b> {erro:.2%}"
                    )
                    row_text.append(hover_item)
                    
                    # Anotação de valor na célula
                    annotations.append({
                        'x': x_clusters[j], 'y': y_descritores[i], 'text': f"{erro:.0%}",
                        'xref': 'x1', 'yref': 'y1', 'showarrow': False,
                        'font': {'color': 'black', 'size': 10}
                    })
                else:
                    hover_item = f"<b>Código:</b> {descritor}<br><b>Habilidade:</b> {descricao}<br>Dados não disponíveis<extra></extra>"
                    row_text.append(hover_item)
            hover_text.append(row_text)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_values, x=x_clusters, y=y_descritores, colorscale='Greens',
            text=hover_text, hoverinfo="text", name='Taxa de Erro',
            hovertemplate="%{text}<extra></extra>" 
        ))
        
        fig_heatmap.update_layout(
            title_text=f"Mapa de Calor: Taxa de Erro por Habilidade e Cluster ({disciplina_selec})",
            yaxis_title='', xaxis_title='Cluster',
            height=max(600, len(y_descritores) * 25),
            annotations=annotations # Adiciona as anotações aqui
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    except Exception as e:
        st.error(f"Não foi possível gerar o mapa de calor. Detalhe: {e}")

def criar_grafico_top10(df_diag_filtrado: pd.DataFrame, disciplina_selec: str):
    """Gera e exibe o gráfico de barras Top 10 Habilidades com Maior Erro."""
    st.subheader("Top 10 Habilidades com Maior Dificuldade")
    st.markdown("O Eixo X mostra o **código da Habilidade**. **Passe o mouse na barra para ver o código e a descrição completa.**")
    
    df_top10 = df_diag_filtrado.groupby(['NU_DESCRITOR_HABILIDADE', 'DESCRICAO_HABILIDADE'])['TAXA_ERRO'].mean()
    df_top10 = df_top10.nlargest(10).reset_index()
    
    fig_top10 = px.bar(
        df_top10,
        x='NU_DESCRITOR_HABILIDADE',
        y='TAXA_ERRO',
        hover_data={
            'NU_DESCRITOR_HABILIDADE': False,   
            'TAXA_ERRO': ':.2%'
        },
        title=f"Top 10 Dificuldades - {disciplina_selec}",
        labels={'TAXA_ERRO': 'Taxa de Erro Média', 'NU_DESCRITOR_HABILIDADE': 'Habilidade (Código)'},
        color_discrete_sequence=[COR_SECUNDARIA_VERDE],
    )
    
    fig_top10.update_traces(
        hovertemplate=
            "<b>Código:</b> %{x}<br>" +
            "<b>Habilidade:</b> %{customdata[0]}<br>" + 
            "<b>Taxa de Erro:</b> %{y}<extra></extra>",
        customdata=df_top10[['DESCRICAO_HABILIDADE']]
    )
    
    fig_top10.update_layout(
        xaxis={'categoryorder':'total descending'} 
    )
    
    fig_top10.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_top10, use_container_width=True)

# --- Interface Principal (Execução) ---

st.title("📊 Painel de Diagnóstico de Habilidades (SAEB)")
st.markdown("Use este painel para analisar o perfil dos alunos e suas dificuldades por habilidade.")

# 1. Filtro Série
st.sidebar.title("Filtros Globais")
serie_selecionada = st.sidebar.selectbox(
    "Selecione a Série",
    ['5EF', '9EF'],
    key='filtro_serie'
)

# Carrega os dados com base na série
dados = carregar_dados(serie_selecionada)

if dados[0] is None:
    st.warning("Não foi possível carregar os dados. Verifique os arquivos e caminhos e reinicie o painel.")
else:
    df_alunos, df_diag_completo = dados
    
    # PEGA AS CONFIGURAÇÕES DA SÉRIE
    config_app = CONFIG_APP_SERIES[serie_selecionada]
    CLUSTER_LEGEND = config_app['CLUSTER_LEGEND']
    CLUSTER_PARA_RISCO = config_app['CLUSTER_PARA_RISCO']
    
    # Variáveis de estado iniciais
    todos_status = sorted(df_alunos['STATUS_RISCO_FINAL'].unique()) 
    todos_clusters = sorted(df_alunos['CLUSTER'].unique())

    if 'filtro_status_risco_global_temp' not in st.session_state:
        # Define os valores iniciais de forma segura
        st.session_state.filtro_status_risco_global_temp = [s for s in STATUS_RISCO_FINAL if s in todos_status]
    if 'filtro_cluster_global_temp' not in st.session_state:
        # Garante que os clusters selecionados inicializados correspondem aos riscos
        set_clusters_from_risco(config_app) # Chamada inicial para sincronizar

    # 2. Filtro de UF 
    ufs_disponiveis = ['Todos os Estados'] + sorted(df_alunos['UF_DESCRICAO'].unique())
    uf_selecionada = st.sidebar.selectbox(
        "Filtrar por Estado (UF)",
        options=ufs_disponiveis,
        key='filtro_uf_global'
    )
    uf_filter_condition = (df_alunos['UF_DESCRICAO'] == uf_selecionada) if uf_selecionada != 'Todos os Estados' else True
    
    # 3. Filtro de Status de Risco 
    st.sidebar.multiselect(
        "Status de Risco",
        [s for s in STATUS_RISCO_FINAL if s in todos_status], 
        default=st.session_state.filtro_status_risco_global_temp,
        key='filtro_status_risco_global_temp',
        on_change=lambda: set_clusters_from_risco(config_app)
    )
    status_selecionados = st.session_state.filtro_status_risco_global_temp
    
    # 4. Filtro de Clusters 
    st.sidebar.multiselect(
        "Clusters de Risco",
        todos_clusters,
        default=st.session_state.filtro_cluster_global_temp,
        key='filtro_cluster_global_temp',
        on_change=lambda: set_risco_from_clusters(STATUS_RISCO_FINAL, config_app)
    )
    clusters_selecionados_global = st.session_state.filtro_cluster_global_temp


    # --- Abas do Painel ---
    tab_visao_geral, tab_diagnostico = st.tabs(
        ["📈 Visão Geral (O Quem)", "🔬 Diagnóstico (O Porquê)"]
    )

    # ======================================================================
    # Aplicar Filtros Globais
    # ======================================================================
    df_alunos_filtrado = df_alunos[
        (df_alunos['STATUS_RISCO_FINAL'].isin(status_selecionados)) &
        (df_alunos['CLUSTER'].isin(clusters_selecionados_global)) & 
        uf_filter_condition
    ].copy() # Adicionado .copy() para evitar SettingWithCopyWarning no Scatter

    # ======================================================================
    # ABA 1: VISÃO GERAL (O QUEM) 
    # ======================================================================
    with tab_visao_geral:
        st.header("Perfil dos Alunos")
        
        if df_alunos_filtrado.empty:
            st.warning("Nenhum aluno encontrado com os filtros selecionados. Verifique os filtros na barra lateral.")
        else:
            
            # 1. KPIs
            criar_kpis_visao_geral(df_alunos_filtrado)
            
            st.subheader("Visualizações")
            gcol1, gcol2 = st.columns(2)

            with gcol1:
                # 2. Gráfico de Pizza de Risco
                criar_grafico_risco(df_alunos_filtrado)
            
            with gcol2:
                # 3. Gráfico de Barras de Cluster
                criar_grafico_cluster(df_alunos_filtrado, CLUSTER_LEGEND)

            st.markdown("---")
            
            # 4. Legenda dos Clusters
            exibir_legenda_clusters(CLUSTER_LEGEND, CLUSTER_PARA_RISCO)

            st.divider()

            # 5. Gráfico de Dispersão (LP vs MT)
            criar_grafico_dispersao(df_alunos_filtrado, CLUSTER_LEGEND)


    # ======================================================================
    # ABA 2: DIAGNÓSTICO (O PORQUÊ)
    # ======================================================================
    with tab_diagnostico:
        st.header("Diagnóstico por Habilidade")
        
        dcol1, dcol2 = st.columns(2)
        
        with dcol1:
            disciplina_selec = st.selectbox(
                "Selecione a Disciplina",
                ['LP', 'MT'],
                key='filtro_disciplina_diag_fixo' 
            )
        with dcol2:
            st.markdown(f"**Clusters em Análise:** {', '.join(clusters_selecionados_global)}")
            clusters_para_diag = clusters_selecionados_global

        
        df_diag_filtrado = df_diag_completo[
            (df_diag_completo['TP_DISCIPLINA'] == disciplina_selec) &
            (df_diag_completo['CLUSTER'].isin(clusters_para_diag))
        ].copy()
        
        # Oculta habilidades específicas se houver configuração
        habilidades_ocultar_disc = HABILIDADES_OCULTAR.get(serie_selecionada, {}).get(disciplina_selec, [])
        if habilidades_ocultar_disc:
            df_diag_filtrado = df_diag_filtrado[
                ~df_diag_filtrado['NU_DESCRITOR_HABILIDADE'].isin(habilidades_ocultar_disc)
            ]

        if df_diag_filtrado.empty:
            st.warning("Nenhum dado de diagnóstico encontrado para os filtros selecionados. Ajuste a seleção de Clusters na barra lateral.")
        else:
            
            st.divider()

            # 1. Mapa de Calor (Heatmap)
            criar_heatmap_habilidade(df_diag_filtrado, CLUSTER_LEGEND, disciplina_selec)

            st.divider()

            # 2. Legenda dos Clusters
            exibir_legenda_clusters(CLUSTER_LEGEND, CLUSTER_PARA_RISCO)

            st.divider()

            # 3. Top 10 Habilidades com Maior Erro
            criar_grafico_top10(df_diag_filtrado, disciplina_selec)

            st.divider()

            # 4. Tabela de Dados Completos da Taxa de Erro (Recolhida)
            st.subheader("Tabela Completa de Diagnóstico por Habilidade")
            
            df_tabela_completa = df_diag_filtrado.groupby(['NU_DESCRITOR_HABILIDADE', 'DESCRICAO_HABILIDADE'])['TAXA_ERRO'].mean().reset_index()
            
            df_tabela_completa.rename(
                columns={'NU_DESCRITOR_HABILIDADE': 'Habilidade', 'DESCRICAO_HABILIDADE': 'Descrição', 'TAXA_ERRO': 'Taxa de Erro Média'},
                inplace=True
            )
            
            # Ordena: Primeiro pela Habilidade (código), depois pela Taxa de Erro
            df_tabela_completa = df_tabela_completa.sort_values(
                by=['Habilidade', 'Taxa de Erro Média'], 
                ascending=[True, False]
            )
            
            df_tabela_completa['Taxa de Erro Média'] = df_tabela_completa['Taxa de Erro Média'].apply(lambda x: f"{x:.2%}")
            
            with st.expander("⬇️ Visualizar Tabela Completa de Habilidades (Todos os Dados)"):
                st.dataframe(
                    df_tabela_completa, 
                    use_container_width=True
                )