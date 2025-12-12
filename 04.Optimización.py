# Verificar que el data frame contenga todas las variables y cálculos necesarios

test_df = df[df['dataset'] == 'test']
desired_dataset = pd.merge(test_df, all_predictions_df[['ds', 'yhat', 'campaign']], on=['ds', 'campaign'], how='left')
desired_dataset['AdCost'] = desired_dataset['budget_USD'] / (desired_dataset['yhat'])

print(desired_dataset.head())

# ----------------------------------------------------------------------------------------------------- #

