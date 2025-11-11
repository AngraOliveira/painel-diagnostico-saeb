def calculate_risk_indicators(self):
    """Calcula indicadores de risco de aprendizagem para ambos os anos"""
    print("\n=== CÁLCULO DE INDICADORES DE RISCO ===")
    
    # Para o 5º EF
    self.df_5ef_clean['MEDIA_PROFICIENCIAS'] = (
        self.df_5ef_clean['PROFICIENCIA_LP'] + self.df_5ef_clean['PROFICIENCIA_MT']
    ) / 2
    
    self.df_5ef_clean['DISCREPANCIA_LP_MT'] = (
        abs(self.df_5ef_clean['PROFICIENCIA_LP'] - self.df_5ef_clean['PROFICIENCIA_MT'])
    )
    
    # Para o 9º EF
    self.df_9ef_clean['MEDIA_PROFICIENCIAS'] = (
        self.df_9ef_clean['PROFICIENCIA_LP'] + self.df_9ef_clean['PROFICIENCIA_MT']
    ) / 2
    
    self.df_9ef_clean['DISCREPANCIA_LP_MT'] = (
        abs(self.df_9ef_clean['PROFICIENCIA_LP'] - self.df_9ef_clean['PROFICIENCIA_MT'])
    )
    
    # Define risco para 5º EF
    limiar_risco_5ef = self.df_5ef_clean['MEDIA_PROFICIENCIAS'].quantile(0.3)
    limiar_discrepancia_5ef = self.df_5ef_clean['DISCREPANCIA_LP_MT'].quantile(0.7)
    
    self.df_5ef_clean['RISCO_APRENDIZAGEM'] = (
        (self.df_5ef_clean['MEDIA_PROFICIENCIAS'] < limiar_risco_5ef) | 
        (self.df_5ef_clean['DISCREPANCIA_LP_MT'] > limiar_discrepancia_5ef)
    ).astype(int)
    
    # Define risco para 9º EF
    limiar_risco_9ef = self.df_9ef_clean['MEDIA_PROFICIENCIAS'].quantile(0.3)
    limiar_discrepancia_9ef = self.df_9ef_clean['DISCREPANCIA_LP_MT'].quantile(0.7)
    
    self.df_9ef_clean['RISCO_APRENDIZAGEM'] = (
        (self.df_9ef_clean['MEDIA_PROFICIENCIAS'] < limiar_risco_9ef) | 
        (self.df_9ef_clean['DISCREPANCIA_LP_MT'] > limiar_discrepancia_9ef)
    ).astype(int)
    
    print(f"Alunos em risco (5º EF): {self.df_5ef_clean['RISCO_APRENDIZAGEM'].sum()}")
    print(f"Taxa de risco (5º EF): {self.df_5ef_clean['RISCO_APRENDIZAGEM'].mean():.2%}")
    print(f"Alunos em risco (9º EF): {self.df_9ef_clean['RISCO_APRENDIZAGEM'].sum()}")
    print(f"Taxa de risco (9º EF): {self.df_9ef_clean['RISCO_APRENDIZAGEM'].mean():.2%}")

def generate_reports(self):
    """Gera relatórios analíticos para ambos os anos"""
    print("\n=== RELATÓRIOS ANALÍTICOS ===")
    
    # Análise por escola - 5º EF
    risco_por_escola_5ef = self.df_5ef_clean.groupby('ID_ESCOLA').agg({
        'RISCO_APRENDIZAGEM': ['count', 'sum', 'mean'],
        'PROFICIENCIA_LP': 'mean',
        'PROFICIENCIA_MT': 'mean'
    }).round(3)
    
    risco_por_escola_5ef.columns = ['TOTAL_ALUNOS', 'ALUNOS_RISCO', 'TAXA_RISCO', 'MEDIA_LP', 'MEDIA_MT']
    risco_por_escola_5ef['TAXA_RISCO_PCT'] = (risco_por_escola_5ef['TAXA_RISCO'] * 100).round(1)
    
    # Análise por escola - 9º EF
    risco_por_escola_9ef = self.df_9ef_clean.groupby('ID_ESCOLA').agg({
        'RISCO_APRENDIZAGEM': ['count', 'sum', 'mean'],
        'PROFICIENCIA_LP': 'mean',
        'PROFICIENCIA_MT': 'mean'
    }).round(3)
    
    risco_por_escola_9ef.columns = ['TOTAL_ALUNOS', 'ALUNOS_RISCO', 'TAXA_RISCO', 'MEDIA_LP', 'MEDIA_MT']
    risco_por_escola_9ef['TAXA_RISCO_PCT'] = (risco_por_escola_9ef['TAXA_RISCO'] * 100).round(1)
    
    print("\nTop 10 escolas com maior taxa de risco (5º EF):")
    print(risco_por_escola_5ef.nlargest(10, 'TAXA_RISCO'))
    
    print("\nTop 10 escolas com maior taxa de risco (9º EF):")
    print(risco_por_escola_9ef.nlargest(10, 'TAXA_RISCO'))
    
    # Salva relatórios
    risco_por_escola_5ef.to_csv('data/processed/risco_por_escola_5ef.csv')
    risco_por_escola_9ef.to_csv('data/processed/risco_por_escola_9ef.csv')
    self.df_5ef_clean.to_csv('data/processed/alunos_5ef_analisados.csv', index=False)
    self.df_9ef_clean.to_csv('data/processed/alunos_9ef_analisados.csv', index=False)
    
    return risco_por_escola_5ef, risco_por_escola_9ef