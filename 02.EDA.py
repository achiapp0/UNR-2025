# Luego de leer lo datos
# Establecer la variable fecha como columna y renombrar como "date"
df = df.reset_index(names="date") 

# Identificar cuál es la respuesta y cual es la variable que refiere al tiempo en la serie
df = df.rename(columns={'date': 'ds', 'nmv_USD': 'y'})

# Formatear la columna 'ds' como datetime
df['ds'] = pd.to_datetime(df['ds'])

# Revisar las primeras filas del dataframe
print(df.head())

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

#------------------------Gráficos-------------------------------#
# Evolución de ventas en dolares ('y') 
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

