# Verificar que el data frame contenga todas las variables y cálculos necesarios

test_df = df[df['dataset'] == 'test']
desired_dataset = pd.merge(test_df, all_predictions_df[['ds', 'yhat', 'campaign']], on=['ds', 'campaign'], how='left')
desired_dataset['AdCost'] = desired_dataset['budget_USD'] / (desired_dataset['yhat'])

print(desired_dataset.head())

# ----------------------------------------------------------------------------------------------------- #

# Optimización general.
# Calcular presupuesto máximo y percentil 10 por campaña y día
max_presupuesto_diario = desired_dataset.groupby(['ds', 'campaign'])['budget_USD'].max().reset_index()
percentil_presupuesto_diario = desired_dataset.groupby(['ds', 'campaign'])['budget_USD'].quantile(0.1).reset_index()
percentil_presupuesto_diario.rename(columns={'budget_USD': 'budget_USD_p10'}, inplace=True)

# Identificar las campañas restringidas al 15%
campañas_restringidas = ['Automotriz',  'Jardineria', 'Mascotas'] #'Menaje', 'Grandes_Electrodomesticos_Cocina',

def optimizar_presupuesto_dia_fijo(df_dia, max_df_dia, p10_df_dia, presupuesto_total_fijo):
    campañas = df_dia['campaign'].values
    yhat = df_dia['yhat'].values
    n = len(campañas)

    presupuestos_max, presupuestos_p10 = [], []

    for camp in campañas:
        max_val = max_df_dia[max_df_dia['campaign'] == camp]['budget_USD'].values[0]
        p10_val = p10_df_dia[p10_df_dia['campaign'] == camp]['budget_USD_p10'].values[0]
        presupuestos_max.append(max_val)
        presupuestos_p10.append(p10_val)

    presupuestos_max = np.array(presupuestos_max)
    presupuestos_p10 = np.array(presupuestos_p10)

    # Definir los límites
    limites_superiores = []
    for i, camp in enumerate(campañas):
        if camp.lower() in campañas_restringidas:
            limites_superiores.append(min(presupuestos_max[i], 0.15 * presupuestos_max[i]))
        else:
            limites_superiores.append(presupuestos_max[i])
    limites_superiores = np.array(limites_superiores)
    limites_inferiores = np.full(n, 10)
    bounds = [(limites_inferiores[i], limites_superiores[i]) for i in range(n)]

    # Inicializar con percentil 10
    x0 = presupuestos_p10

    # Definir la Función objetivo. Minimizar adcost total
    def objetivo(presupuestos):
        return np.sum(presupuestos / (yhat + 1e-8))

    # Agregar restricciones
    indices_no_restringidas = [i for i, c in enumerate(campañas) if c.lower() not in campañas_restringidas]

    def restriccion_presupuesto_no_restringidas(presupuestos):
        suma_no_restringidas = np.sum(presupuestos[indices_no_restringidas])
        max_total_no_restringidas = 0.75 * presupuesto_total_fijo
        return max_total_no_restringidas - suma_no_restringidas

    def restriccion_presupuesto_total(presupuestos):
        return presupuesto_total_fijo - np.sum(presupuestos)

    restricciones = [
        {'type': 'ineq', 'fun': restriccion_presupuesto_no_restringidas},
        {'type': 'ineq', 'fun': restriccion_presupuesto_total},
        {'type': 'ineq', 'fun': lambda x: yhat - 1e-6}  # yhat > 0
    ]

    # Optimizar la función
    resultado = minimize(objetivo, x0, bounds=bounds, constraints=restricciones, method='SLSQP')

    # Gaurdar los resultados
    resultados = []
    for i, camp in enumerate(campañas):
        optimizado = resultado.x[i] if resultado.success else np.nan
        adcost = optimizado / yhat[i] if yhat[i] > 0 and resultado.success else np.nan
        resultados.append({
            'ds': df_dia['ds'].iloc[0],
            'campaign': camp,
            'yhat': yhat[i],
            'budget_USD_max': presupuestos_max[i],
            'budget_USD_p10': presupuestos_p10[i],
            'budget_USD_optimizado': optimizado,
            'adcost_optimizado': adcost,
            'success': resultado.success,
            'message': resultado.message
        })
    return pd.DataFrame(resultados)

# Ejecutar la optimización
resultados_optim = optimizar_presupuesto_diario_por_campana(desired_dataset, max_presupuesto_diario_por_campana)
print(resultados_optim)

# Mostrar los resultados por campaña
desired_dataset['adcost'] = desired_dataset['budget_USD'] / (desired_dataset['yhat'] + 1e-8)

resumen_original = desired_dataset.groupby('campaign').agg({
    'budget_USD': 'sum',
    'yhat': 'sum',
    'adcost': 'mean'
}).rename(columns={
    'budget_USD': 'budget_total',
    'yhat': 'yhat_total',
    'adcost': 'adcost_promedio'
}).reset_index()

resumen_optimizado = resultados_optim.groupby('campaign').agg({
    'budget_USD_optimizado': 'sum',
    'adcost_optimizado': 'mean'
}).rename(columns={
    'budget_USD_optimizado': 'budget_optimizado_total',
    'adcost_optimizado': 'adcost_optimizado_promedio'
}).reset_index()

resumen_completo = resumen_original.merge(resumen_optimizado, on='campaign', how='outer')

print(resumen_completo)

# Identificar la cantidad de éxitos en la optimización
cantidad_exitos = resultados_optim['success'].sum()
print(f'Cantidad de optimizaciones exitosas: {cantidad_exitos}')

total_registros_dataset = len(desired_dataset.groupby(['ds','campaign']))
print(f"Cantidad de (fecha, campaña) únicas en dataset: {total_registros_dataset}")

# ----------------------------------------------------------------------------------------------------- #
# Optimización para un día dado y un monto fijo de presupuesto. Es un ejemplo de los planteados en el trabajo final

# Setear un día específico (día de cyber) y un presupuesto fijo
fecha_objetivo = '2023-06-28'  # Cambiar por la fecha deseada
presupuesto_total_fijo = 10000  # Cambiar por el presupuesto total deseado para ese día

# Filtrar y ejecutar
df_dia = desired_dataset[desired_dataset['ds'] == fecha_objetivo]
max_dia = max_presupuesto_diario[max_presupuesto_diario['ds'] == fecha_objetivo]
p10_dia = percentil_presupuesto_diario[percentil_presupuesto_diario['ds'] == fecha_objetivo]

resultado_dia_fijo = optimizar_presupuesto_dia_fijo(df_dia, max_dia, p10_dia, presupuesto_total_fijo)

# Verificar y mostrar los resultados
print(resultado_dia_fijo)
print("Suma total optimizada:", resultado_dia_fijo['budget_USD_optimizado'].sum())
print("Presupuesto total fijado:", presupuesto_total_fijo)

# ----------------------------------------------------------------------------------------------------- #
# Graficar Presupuesto y AdCost originales versus optimizados

# Preparar data frame para presupuesto
merged_df = pd.merge(resultados_optim, df[['ds', 'campaign', 'budget_USD']], on=['ds', 'campaign'], how='left')
filtered_df = merged_df[merged_df['success']]
daily_sums = filtered_df.groupby('ds')[['budget_USD', 'budget_USD_optimizado']].sum().reset_index()
print(daily_sums)

# Gráfico general de evolución de presupuesto para todas las campañas juntas
plt.figure(figsize=(12, 6))
plt.plot(daily_sums['ds'], daily_sums['budget_USD'], label='Original Budget')
plt.plot(daily_sums['ds'], daily_sums['budget_USD_optimizado'], label='Optimized Budget')
plt.title("Total Daily Budget Comparison")
plt.xlabel("Date")
plt.ylabel("Total Budget (USD)")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico de evolución de presupuesto por campaña
for campaign in campaigns:
      campaign_data = filtered_df[filtered_df['campaign'] == campaign]
  
    daily_sums_campaign = campaign_data.groupby('ds')[['budget_USD', 'budget_USD_optimizado']].sum().reset_index()
    plt.figure(figsize=(12, 6))  
    plt.plot(daily_sums_campaign['ds'], daily_sums_campaign['budget_USD'], label='Original Budget', color='blue')
    plt.plot(daily_sums_campaign['ds'], daily_sums_campaign['budget_USD_optimizado'], label='Optimized Budget', color='orange')
    plt.title(f"Total Daily Budget Comparison - Campaign: {campaign}")
    plt.xlabel("Date")
    plt.ylabel("Total Budget (USD)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show() 

# Preparar data frame para AdCost
merged_df = pd.merge(resultados_optim, test_df[['ds', 'campaign', 'adcost', 'y']], on=['ds', 'campaign'], how='left')
filtered_df = merged_df[merged_df['success']]
filtered_df['adcost_optimizado'] = filtered_df['budget_USD_optimizado'] / filtered_df['y']
daily_adcost_sums = filtered_df.groupby('ds')[['adcost', 'adcost_optimizado']].sum().reset_index()
print(daily_adcost_sums)

# Gráfico general de evolución de AdCost para todas las campañas juntas
plt.figure(figsize=(12, 6))
plt.plot(daily_adcost_sums['ds'], daily_adcost_sums['adcost'], label='Original Adcost')
plt.plot(daily_adcost_sums['ds'], daily_adcost_sums['adcost_optimizado'], label='Optimized Adcost')
plt.title("Total Daily Adcost Comparison")
plt.xlabel("Date")
plt.ylabel("Total Adcost")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico de evolución de AdCost por campaña
filtered_df['adcost_optimizado'] = filtered_df['budget_USD_optimizado'] / filtered_df['y']
filtered_df['adcost'] = filtered_df['budget_USD'] / filtered_df['y']
campaigns = filtered_df['campaign'].unique()

for campaign in campaigns:
    campaign_data = filtered_df[filtered_df['campaign'] == campaign]

    plt.figure(figsize=(12, 6))
    plt.plot(campaign_data['ds'], campaign_data['adcost'], label='Original Adcost', color='blue')
    plt.plot(campaign_data['ds'], campaign_data['adcost_optimizado'], label='Optimized Adcost', color='orange')
    plt.title(f"Campaign: {campaign}")
    plt.xlabel("Date")
    plt.ylabel("Adcost")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()  

#--------------------------------------------------------------------------------------------------------------------------#
# Prueba de Wilcoxon para evaluar si la optimización tiene significación estadística

# Calcular el retorno por unidad de gasto para original y optimizado
filtered_df['adcost_optimizado'] = filtered_df['budget_USD_optimizado'] / filtered_df['y']
filtered_df['adcost'] = filtered_df['budget_USD'] / filtered_df['y']   # Real: Ingreso (y) / Costo (adcost)

# Calcular la diferencia de retorno (optimizado vs real)
filtered_df['adcost_diff'] = filtered_df['adcost_optimizado'] - filtered_df['adcost']

# Agrupar por campaña para obtener la media del retorno y la diferencia
campaign_returns = filtered_df.groupby('campaign')[['adcost', 'adcost_optimizado', 'adcost_diff']].mean().reset_index()

# Mostrar los resultados por campaña
print(campaign_returns)

# Guardar los resultados del test de Wilcoxon
wilcoxon_results = []

# Agrupar por campaña
for campaign, campaign_data in filtered_df.groupby('campaign'):
    # Verificar que haya más de una observación para aplicar el test
    if len(campaign_data) < 2:
        continue  # Salir si no hay suficientes datos

    # Aplicar el test de Wilcoxon
    w_stat, p_value = wilcoxon(campaign_data['return_real'], campaign_data['return_opt'])

    # Guardar los resultados
    wilcoxon_results.append({
        'campaign': campaign,
        'w_statistic': w_stat,
        'p_value': p_value,
        'significativo': p_value < 0.05
    })

# Convertir a DataFrame para mostrar los resultados
df_wilcoxon_results = pd.DataFrame(wilcoxon_results)

# Mostrar resultados
print(df_wilcoxon_results)
#--------------------------------------------------------------------------------------------------------------------------#
