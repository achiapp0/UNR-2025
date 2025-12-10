# Luego de leer lo datos
# Establecer la variable fecha como columna y renombrar como "date"
df = df.reset_index(names="date") 

#----------------------------------------------------------------------------------------------------------------------------------------------#
# Identificar cuál es la respuesta y cual es la variable que refiere al tiempo en la serie
df = df.rename(columns={'date': 'ds', 'nmv_USD': 'y'})

# Formatear la columna 'ds' como datetime
df['ds'] = pd.to_datetime(df['ds'])

# Revisar las primeras filas del dataframe
print(df.head())
#---------------------------------------------------------------------------------------------------------------------------------------------#

# Definir el diccionario de variables con los nombres que se utilizarán para visualización
nombres_para_visualizacion = {
    'ds':'Fecha',
    'campaign': 'Campaña',
    'is_cyber': 'Cyber',
    'is_outlier': 'Outlier',
    'y': 'Venta neta (en dólares)',
    'clicks': 'Nro clicks hechos sobre el anuncio',
    'impressions': 'Cantidad de veces que se muestra el anuncio',
    'visitors': 'Nro visitantes',
    'visits': 'Nro Visitas',
    'orders': 'Compras',
    'total_units_net': 'Total unidades vendidas',
    'budget_USD': 'Presupuesto/inversión en GAds',
    'cost_USD': 'Costo facturado en Google Ads'
}
#----------------------------------------------------------------------------------------------------------------------------------------------#

# Identificar datos faltantes por campaña

# Agrupar datos y obtener fechas únicas para cada campaña
campaign_dates = df.groupby('campaign')['ds'].unique()

# Obtener todas las fechas únicas de todas las campañas
all_dates = pd.Series(df['ds'].unique()).sort_values()

# Iterar a través de las campañas y encontrar las fechas faltantes
for campaign, dates in campaign_dates.items():
    missing_dates = all_dates[~all_dates.isin(dates)]
    if not missing_dates.empty:
        print(f"Campaign: {campaign}")
        print(f"Missing Dates: {missing_dates.tolist()}")
#----------------------------------------------------------------------------------------------------------------------------------------------#
        
# Identificación de outliers por campaña

# Definir el guardado de los resultados
outlier_results = []

# Iterar a través de cada campaña
for campaign in campaigns:
    print(f"\nDetecting outliers for campaign: {campaign}")

    df_campaign = df[df['campaign'] == campaign].copy()

    # Seleccionar las variables sobre las que se identificarán los outliers
    features_for_copod = ['y', 'budget_USD', 'clicks', 'impressions', 'visitors', 'visits', 'orders', 'total_units_net', 'cost_USD'] 

    campaign_features = df_campaign[features_for_copod]
    campaign_features_cleaned = campaign_features.dropna()
    original_indices = campaign_features_cleaned.index

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(campaign_features_cleaned)

    # Inicialziar y entrenar el modelo COPOD para identificar outliers
    clf = COPOD()
    clf.fit(X_scaled)

    # Identificar, etiquetar y calcular score de outliers
    outlier_labels = clf.labels_
    decision_scores = clf.decision_scores_

    # Guardar los resultados en forma temporaria por campaña
    temp_results_df = pd.DataFrame({
        'original_index': original_indices,
        'copod_outlier': outlier_labels,
        'copod_score': decision_scores
    })

    # Abrir los resultados
    outlier_results.append(temp_results_df)

# Concatenar los resultados para todas las campañas
all_outlier_results = pd.concat(outlier_results)

# Unir los resultados con el dataframe original

df = df.merge(all_outlier_results, left_index=True, right_on='original_index', how='left')

# Eliminar 'original_index' temporaria
df = df.drop(columns=['original_index'])

# Completar NaNs potenciales en 'copod_outlier' y 'copod_score'
df['copod_outlier'] = df['copod_outlier'].fillna(0).astype(int)
df['copod_score'] = df['copod_score'].fillna(0)

# Validar que el proceso haya sido correcto observando el data frame
print("\nDataFrame with COPOD Outlier Detection Results:")
print(df.head())
print("\nCOPOD Outlier Count per Campaign:")
print(df.groupby('campaign')['copod_outlier'].value_counts())
#----------------------------------------------------------------------------------------------------------------------------------------------#

# Calcular las medidas estadísticas descriptivas de las variables numéricas
descriptive_stats = df[['y', 'clicks', 'impressions', 'visitors', 'visits', 'orders', 'total_units_net', 'budget_USD', 'cost_USD']].describe()
descriptive_stats_renamed = descriptive_stats.rename(index=nombres_para_visualizacion)
descriptive_stats_renamed = descriptive_stats_renamed.rename(columns={'mean': 'Promedio', 'std': 'Desviación E.'})

# Aplicar estilo y formato de visualización del output
styled_stats = descriptive_stats_display.style.format("{:,.0f}") \
                                    .set_caption("Descriptive Statistics") \
                                    .background_gradient(cmap='Blues') 

display(styled_stats)

# Calcular las medidas estadísticas descriptivas de las variables numéricas para cada Campaña

variables_to_describe = ['y', 'clicks', 'impressions', 'visitors', 'visits', 'orders', 'total_units_net', 'budget_USD', 'cost_USD']

print("Descriptive Statistics by Campaign:")

# Agrupar por 'campaign' las medidas descriptivas
for campaign, group_df in df.groupby('campaign'):
    print(f"\n--- Campaign: {campaign} ---")

    descriptive_stats_campaign = group_df[variables_to_describe].describe()
    descriptive_stats_campaign_display = descriptive_stats_campaign.rename(columns=nombres_para_visualizacion)
    styled_stats_campaign = descriptive_stats_campaign_display.style \
        .format(
            "{:,.0f}")\
        .set_caption(f"Descriptive Statistics for Campaign: {campaign}") 
        .background_gradient(cmap='Blues')

    # Mostrar los resultados por campaña
    display(styled_stats_campaign)

print("\n--- End of Descriptive Statistics by Campaign ---")

# Ordenar por fecha
df = df.sort_values(by='ds')

#Verificar las primeras filas para confirmar el orden
print(df.head())

# Calcular las métricas CRT, ROI, AdCost para conocer el desempeño general y por campaña

# Para todas las campañas juntas
total_row = df[['visits', 'visitors', 'y', 'cost_USD', 'budget_USD']].sum().to_frame().T

# Definir el cálculo de las 3 métricas para todas las campañas juntas
total_row['campaign'] = 'Total'
total_row['CRT'] = total_row['visits'] / (total_row['visitors'] )
total_row['ROI'] = total_row['y'] / (total_row['cost_USD'] )
total_row['AdCost'] = total_row['budget_USD'] / (total_row['y'] )

# Ordenar resultados totales
total_row = total_row[['campaign', 'CRT', 'ROI', 'AdCost']]

# Calcular métricas CRT, ROI, AdCost para evaluar cada campaña
campaign_summary = df.groupby('campaign')[['visits', 'visitors', 'y', 'cost_USD', 'budget_USD']].sum().reset_index()

# Definir el cálculo de las 3 métricas
epsilon = 1e-9 # Valor mínimo para evitar la división por 0
campaign_summary['CRT'] = campaign_summary['visits'] / (campaign_summary['visitors'])
campaign_summary['ROI'] = campaign_summary['y'] / (campaign_summary['cost_USD'] )
campaign_summary['AdCost'] = campaign_summary['budget_USD'] / (campaign_summary['y'])

# Ordenar resultados por campaña
final_campaign_summary_with_total = pd.concat([campaign_summary[['campaign', 'CRT', 'ROI', 'AdCost']], total_row], ignore_index=True)

# Mostrar resultados por campaña y total
print("Campaign Performance Summary (CRT, ROI, AdCost) with Total:")
print(final_campaign_summary_with_total)

#--------------------------------------------------------------------------------------------------------#
#----------------Gráficos a modo de ejemplo se presentan para la variable respuesta 'y'------------------#
#--------------------------------------------------------------------------------------------------------#

#  ------------------------------------------------- Evolución de ventas en dolares ('y')  -------------------------------------------------#
if 'year_month' not in df.columns:
    df['year_month'] = df['ds'].dt.to_period('M')

# Calcular la suma total de ventas ('y') 
sales_by_month = df.groupby('year_month')['y'].sum().reset_index()

# Convertir 'year_month' a datetime 
sales_by_month['year_month'] = sales_by_month['year_month'].dt.to_timestamp()

try:
    y_visual_name = nombres_para_visualizacion.get('y', 'Ventas Totales')
except NameError:
    y_visual_name = 'Ventas Totales' # Fallback

# Crear el gráfico de línea para las ventas totales
plt.figure(figsize=(15, 7))
total_sales = sales_by_month.sort_values(by='year_month')
if len(total_sales) > 1:
    x_total = mdates.date2num(total_sales['year_month'])
    y_total = total_sales['y']

    try:
        # Crear un 'spline' de interpolación cúbica
        spl_total = make_interp_spline(x_total, y_total, k=3)
        x_smooth_total = np.linspace(x_total.min(), x_total.max(), 200)
        y_smooth_total = spl_total(x_smooth_total)
        
        # Convertir timestamp numérico a datetime
        x_smooth_total_dt = mdates.num2date(x_smooth_total)
        
        # Graficar la línea suavizada
        plt.plot(x_smooth_total_dt, y_smooth_total, linestyle='-', color='darkred', label=y_visual_name)
        # Graficar los puntos originales de los datos
        plt.plot(total_sales['year_month'], total_sales['y'], 'o', color='darkred', markersize=4)
        
    except ValueError as e:
        # Fallback si el suavizado falla (ej. pocos puntos)
        print(f"No se pudo crear el 'spline' suave: {e}. Graficando línea original.")
        plt.plot(total_sales['year_month'], total_sales['y'], marker='o', linestyle='-', color='darkred', label=y_visual_name, markersize=4)

else:
    # Si hay un solo punto de dato o menos
    plt.plot(total_sales['year_month'], total_sales['y'], marker='o', linestyle='-', color='darkred', label=y_visual_name, markersize=4)

# Configurar título y etiquetas
plt.title(f"Evolución de {y_visual_name} Agrupadas por Mes") # Título ajustado
plt.xlabel("Mes")
plt.ylabel(f"{y_visual_name}")

# Mostrar todas las etiquetas de meses en el eje X
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45, ha='right')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 

#  ------------------------------------------------- Evolución de ventas en dolares ('y') por campaña -------------------------------------------------#
campaigns = df['campaign'].unique()

# Definir la grilla 
n_campaigns = len(campaigns)
cols = 3 
rows = (n_campaigns + cols - 1) // cols  
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)  
axes = axes.flatten()  

# Crear los gráficos de serie de tiempo en la grilla por campaña
for i, campaign in enumerate(campaigns):
    ax = axes[i]  # Get the current subplot
    campaign_data = df[df['campaign'] == campaign]
    ax.plot(campaign_data['ds'], campaign_data['budget_USD'], color='skyblue')
    y_visual_name = nombres_para_visualizacion.get('y', 'y') 
    
    ax.set_title(f"{campaign}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(y_visual_name)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True) 

# Eliminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

# ------------------------------------------------ Histograma de ventas en dolares ('y') por campaña -------------------------------------------------#

campaigns = df['campaign'].unique()

# Definir la grilla 
n_campaigns = len(campaigns)
cols = 3 
rows = (n_campaigns + cols - 1) // cols  
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False) 
axes = axes.flatten() 

# Crear los histogramas en la grilla por campaña
for i, campaign in enumerate(campaigns):
    ax = axes[i] 
    campaign_data = df[df['campaign'] == campaign]

    sns.histplot(data=campaign_data, x='y', ax=ax, kde=True, color='royalblue') 
    
    y_visual_name = nombres_para_visualizacion.get('y', 'y') 

    ax.set_title(f"{campaign}")
    ax.set_xlabel(y_visual_name)
    ax.set_ylabel("Frecuencia") 
    
# Eliminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()


#  ------------------------------------------------- Box-plot de ventas en dolares ('y') por campaña  -------------------------------------------------#

campaigns = df['campaign'].unique()

# Definir la grilla 
n_campaigns = len(campaigns)
cols = 3 
rows = (n_campaigns + cols - 1) // cols  
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False) 
axes = axes.flatten() 

# Crear los boxplot en la grilla por campaña
for i, campaign in enumerate(campaigns):
    ax = axes[i] 
    campaign_data = df[df['campaign'] == campaign]

    sns.boxplot(data=campaign_data, x='y', ax=ax, kde=True, color='skyblue') 
    
    y_visual_name = nombres_para_visualizacion.get('y', 'y') 

    ax.set_title(f"{campaign}")
    ax.set_xlabel(y_visual_name)
    ax.set_ylabel("Frecuencia") 
    
# Eliminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

#  ------------------------------------------------- Evolución de ventas en dolares ('y') por campaña identificando Cyber Day y No Cyber Day -------------------------------------------------#
# Es un gráfico a modo ejemplo, se ajustó para otras variables como día de semana / fines de semana, Evento / No evento, Outliers/No outliers

df['cyber_status'] = df['is_cyber'].apply(lambda x: 'Cyber Day' if x == 1 else 'No Cyber Day')

campaigns = df['campaign'].unique()

# Definir la griolla
n_campaigns = len(campaigns)
cols = 3 
rows = (n_campaigns + cols - 1) // cols  

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False) 
axes = axes.flatten()  

y_visual_name = nombres_para_visualizacion.get('y', 'y')

# Crear el gráfico de series de tiempo para la grilla
for i, campaign in enumerate(campaigns):
    ax = axes[i]  # Get the current subplot
    campaign_data = df[df['campaign'] == campaign].copy() 

    if 'cyber_status' not in campaign_data.columns:
        campaign_data['cyber_status'] = campaign_data['is_cyber'].apply(lambda x: 'Cyber Day' if x == 1 else 'No Cyber Day')

    sns.lineplot(data=campaign_data, x='ds', y='y', hue='cyber_status', ax=ax,
                 palette={'No Cyber Day': 'royalblue', 'Cyber Day': 'red'}, marker='o', markersize=3, linestyle='')

    ax.set_title(f"{campaign}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(y_visual_name)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True) 
    ax.legend(title='Tipo de Día', loc='upper right') 

# Eliminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

#  ------------------------------------------------- Boxplot de ventas en dolares ('y') por campaña identificando Cyber Day y No Cyber Day -------------------------------------------------#
# Es un gráfico a modo ejemplo, se ajustó para otras variables como día de semana / fines de semana, Evento / No evento

df['cyber_status'] = df['is_cyber'].apply(lambda x: 'Cyber Day' if x == 1 else 'No Cyber Day')

if 'cyber_status' not in df.columns:
    df['cyber_status'] = df['is_cyber'].apply(lambda x: 'Cyber Day' if x == 1 else 'No Cyber Day')

campaigns = df['campaign'].unique()

# Definir la grilla
n_campaigns = len(campaigns)
cols = 3  
rows = (n_campaigns + cols - 1) // cols  

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)  
axes = axes.flatten()  

y_visual_name = nombres_para_visualizacion.get('y', 'y')

# Crear el box plot por campaña en la grilla, separado por estado cyber day / no cyber day
for i, campaign in enumerate(campaigns):
    ax = axes[i] 
    campaign_data = df[df['campaign'] == campaign].copy()

    sns.boxplot(data=campaign_data, x='cyber_status', y='y', ax=ax,
                palette={'Cyber Day': 'pink', 'No Cyber Day': 'royalblue'})

    ax.set_title(f"{campaign}")
    ax.set_xlabel("Tipo de Día (Cyber vs Normal)")
    ax.set_ylabel(y_visual_name)
    ax.tick_params(axis='x', rotation=45)

    ax.grid(True, axis='y') 

# Eliminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

#  ------------------------------------------------- Cálculo de medidas de rendimiento de campaña identificando Cyber Day y No Cyber Day  -------------------------------------------------#
# Es una tabla a modo ejemplo, se ajustó para otras variables como día de semana / fines de semana, Evento / No evento

df['cyber_status'] = df['is_cyber'].apply(lambda x: 'Cyber Day' if x == 1 else 'No Cyber Day')

# Agrupar por campaña y tipo de día
grouped = df.groupby(['campaign', 'cyber_status'])[['visits', 'visitors', 'y', 'cost_USD', 'budget_USD']].sum().reset_index()

# Cálculo de las 3 métricas
epsilon = 1e-9
grouped['CRT'] = grouped['visits'] / (grouped['visitors'] + epsilon)
grouped['ROI'] = grouped['y'] / (grouped['cost_USD'] + epsilon)
grouped['AdCost'] = grouped['budget_USD'] / (grouped['y'] + epsilon)

# Calcular el total por tipo de día
total = df.groupby('cyber_status')[['visits', 'visitors', 'y', 'cost_USD', 'budget_USD']].sum().reset_index()
total['campaign'] = 'Total'
total['CRT'] = total['visits'] / (total['visitors'] + epsilon)
total['ROI'] = total['y'] / (total['cost_USD'] + epsilon)
total['AdCost'] = total['budget_USD'] / (total['y'] + epsilon)

# Unir campañas + total
summary_all = pd.concat([grouped, total], ignore_index=True)

# Mantener solo las columnas necesarias
summary_all = summary_all[['campaign', 'cyber_status', 'CRT', 'ROI', 'AdCost']]

# Crear y ordenar la tabla pivoteada (doble entrada)
pivot_table = summary_all.pivot(index='campaign', columns='cyber_status', values=['CRT', 'ROI', 'AdCost'])
pivot_table = pivot_table[[('CRT', 'Cyber Day'), ('CRT', 'No Cyber Day'),
                        ]]
pivot_table = pivot_table.round(3)

# Mostrar la tabla
print("Tabla de rendimiento por campaña (Cyber Day vs No Cyber Day):")
print(pivot_table)

# ------------------------------------------------- Matriz de correlación -------------------------------------------------#
# Seleccionar las variables numéricas para la matriz de correlación
variables_numericas = ['y', 'clicks', 'impressions', 'visitors', 'visits', 'orders',
                       'total_units_net', 'budget_USD', 'cost_USD']

# Crear un sub-dataframe con solo las variables numéricas
df_numerico = df[variables_numericas]

# Calcular la matriz de correlación
matriz_correlacion = df_numerico.corr()

# Renombrar los índices y columnas de la matriz de correlación
matriz_correlacion_renombrada = matriz_correlacion.rename(
    index=nombres_para_visualizacion,
    columns=nombres_para_visualizacion
)

# Visualizar la matriz de correlación con un mapa de calor usando los nombres renombrados
plt.figure(figsize=(14, 12)) 
sns.heatmap(matriz_correlacion_renombrada, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Mapa de Calor de la Matriz de Correlación")
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)
plt.tight_layout() 
plt.show()

#------------------------------------- Definición de componente HOLIDAY -------------------------------------------------#
# Definir las variables clave a verificar para datos faltantes
columnas_clave = ['y', 'clicks', 'impressions', 'visitors', 'visits', 'orders',
                  'total_units_net', 'budget_USD', 'cost_USD']

# Identificar is_cyber es 1, is_outlier es 1 o is_event es 1
condicion_eventos = (df['is_cyber'] == 1) | (df['is_outlier'] == 1) | (df['is_event'] == 1)

# Identificar datos faltantes en alguna de las variables clave
condicion_missing = df[columnas_clave].isnull().any(axis=1)

# Definir la variable 'holiday': es 1 si se cumple la condición_eventos O la condicion_missing
df['holiday'] = np.where(condicion_eventos | condicion_missing, 1, 0)

# Verificar las primeras filas para ver la nueva columna 'holiday'
print("\nDataFrame con la nueva columna 'holiday':")
print(df.head())

# Contar cuántas filas tienen holiday = 1
print("\nConteo de filas con holiday = 1:")
print(df['holiday'].value_counts())

# Definir el data frame para la componente Holiday
holidays_df = df.loc[df['holiday'] != '', ['holiday', 'ds']].copy()
