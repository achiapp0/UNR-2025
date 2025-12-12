# Comenzar con la separación del conjunto de datos en train y test antes de ajustar el modelo
def train_test_per_campaigns(df, train_frac):
    '''
    Crete the train-test datasets for each campaign.
        Parameters:
            df (DataFrame): Table with campaign's historical data.
            train_frac (float): Test fraction for the split.
        Returns:
            df (DataFrame): Table with campaign's historical data labeled
                with the "dataset" column.
    '''
    # Flag train-test
    d_list = []
    for campaign in df['campaign'].unique():
        df_tmp = df[(df['campaign'] == campaign)]
        df_tmp.reset_index(drop=True, inplace=True)
        itrain = int(train_frac * df_tmp.shape[0])
        df_train, df_test =  df_tmp[:itrain].copy(), df_tmp[itrain+3:].copy()
        df_train['dataset'], df_test['dataset'] = 'train', 'test'
        d_list.append(pd.concat([df_train, df_test]))
    df = pd.concat(d_list)
    return df.reset_index(drop=True)

# Correr la función y validar la cantidad de registros en cada conjunto
train_frac = 0.8
df = train_test_per_campaigns(df, train_frac)
print('Número de registros en cada data set', df['dataset'].value_counts())

# Graficar las series de las variables de interés identificando los datos de train y de test
# Ordenar el conjunto de datos
df = df.sort_values(by='ds')

# Identificar las variables de interés
variables_a_graficar = ['y', 'clicks', 'visitors', 'visits', 'orders', 'total_units_net', 'impressions', 'budget_USD', 'cost_USD']

# Obtener el nombre para visualización de la variable actual
var_visual_name = nombres_para_visualizacion.get(var, var)

# Definir la grilla
cols = 3
rows = (len(variables_a_graficar) + cols - 1) // cols 
fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5)) 
axs = axs.flatten() 

sns.set_style('whitegrid')

# Fecha de división entre entrenamiento  (train) y prueba (test)
split_date = pd.to_datetime('2023-05-01')

for i, var in enumerate(variables_a_graficar):
    ax = axs[i] 

    df_ = df.copy()
    df_['ds'] = pd.to_datetime(df_['ds']) 

    # Separar los datos en conjuntos de entrenamiento y prueba
    df1 = df_[df_.dataset == 'train'].set_index('ds')
    df2 = df_[df_.dataset == 'test'].set_index('ds')


    ax.set_title(var_visual_name, fontsize=14, loc='left')
    if var in df1.columns and var in df2.columns:
        ax.plot(df1.index, df1[var], label='Training', color='silver')
        ax.plot(df2.index, df2[var], label='Test', color='royalblue')
        ax.axvline(split_date, color='darkgray', ls='--')
        ax.legend(['Training Set', 'Test Set'])
    else:
        ax.text(0.5, 0.5, f'Columna "{var}" no encontrada', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
        ax.set_title(f"{var_visual_name} (Error)", fontsize=14, loc='left', color='red') 

    # Configurar etiquetas de los ejes 
    ax.set_xlabel('Fecha', labelpad=20) 
    ax.set_ylabel(var_visual_name, labelpad=20) 
    ax.tick_params(axis='x', rotation=45)
    ax.grid(False) 

# Eliminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------#
# -------------------------------------------- Ajustar Modelo Prophet ------------------------------------------#
# --------------------------------------------------------------------------------------------------------------#

# Revisar y validar el data frame Holidays
print("Inspecting holidays_df:")
print(holidays_df.head())
print(holidays_df['holiday'].dtype)

# Setear el formato de Holiday
if holidays_df['holiday'].dtype != 'object':
    print("Converting 'holiday' column to string type.")
    holidays_df['holiday'] = holidays_df['holiday'].astype(str)
    print("Conversion complete. New dtype:", holidays_df['holiday'].dtype)

# Identificar las covariables a incluir en el modelo
covariables = ['clicks', 'impressions', 'visitors', 'visits', 'orders', 'total_units_net',
               'budget_USD', 'cost_USD']

# Listar las campañas
campaigns = df['campaign'].unique()

# Preparar diccionarios para modelos, predicciones y métricas
models = {}
forecasts = {}
metrics = {}
all_predictions = []

for campaign in campaigns:
    print(f"\n🔧 Entrenando modelo para campaña: {campaign}")

    # Filtrar datos por campaña
    df_campaign = df[df['campaign'] == campaign].copy()

    # Separar Train/Test (80/20)
    train_size = int(len(df_campaign) * 0.8)
    train_df = df_campaign.iloc[:train_size].copy()
    test_df = df_campaign.iloc[train_size:].copy()

    # Configurar modelo Prophet
    model = Prophet(
        growth='linear',
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto',
        seasonality_mode='multiplicative',
        seasonality_prior_scale=6.0,
        holidays=holidays_df, # Use the potentially corrected holidays_df
        holidays_prior_scale=8.0,
        holidays_mode='multiplicative',
        n_changepoints=100,
        changepoint_range=0.6, #modelar con 0.8
        changepoint_prior_scale=0.9,
        interval_width=0.95
    )

    # Agregar las covariables/regresores
    for cov in covariables:
        model.add_regressor(cov)

    # Entrenar el modelo
    model.fit(train_df[['ds', 'y'] + covariables])
    models[campaign] = model

    # Crear DataFrame para pronósticos
    future = test_df[['ds'] + covariables].copy()
    forecast = model.predict(future)
    forecasts[campaign] = forecast

    # Agregar valores reales
    forecast['campaign'] = campaign
    forecast['y'] = test_df['y'].values
    all_predictions.append(forecast[['ds', 'y', 'yhat', 'campaign']])

    # Calcular métricas de bondad de ajuste
    y_train_pred = model.predict(train_df[['ds'] + covariables])['yhat'].values
    y_train_real = train_df['y'].values
    y_test_pred = forecast['yhat'].values
    y_test_real = test_df['y'].values


    mae_train = mean_absolute_error(y_train_real, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred))
    mse_train = mean_squared_error(y_train_real, y_train_pred)
    mape_train = np.mean(np.abs((y_train_real - y_train_pred) / np.where(y_train_real == 0, 1e-8, y_train_real))) * 100
    #smape_train = smape(y_train_real, y_train_pred)
    #mdape_train = mdape(y_train_real, y_train_pred)

    mae_test = mean_absolute_error(y_test_real, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_real, y_test_pred))
    mse_test = mean_squared_error(y_test_real, y_test_pred)
    mape_test = np.mean(np.abs((y_test_real - y_test_pred) / np.where(y_test_real == 0, 1e-8, y_test_real))) * 100
    #smape_test = smape(y_test_real, y_test_pred)
   #mdape_test = mdape(y_test_real, y_test_pred)

    metrics[campaign] = {
        "Train": {"MAE": mae_train, "RMSE": rmse_train, "MAPE": mape_train,  "MSE": mse_train, "Observations": len(train_df)},
        "Test": {"MAE": mae_test, "RMSE": rmse_test, "MAPE": mape_test, "MSE": mse_test, "Observations": len(test_df)}
    }

    # Graficar y guardar componentes
    print(f"📊 Componentes del modelo para campaña: {campaign}")
    fig = model.plot_components(forecast)
    plt.close(fig)

# --------------------------------------------------------------------------------------------------------------#
# Combinar predicciones
all_predictions_df = pd.concat(all_predictions)

# Mostrar resumen de métricas
print("\n✅ Resultados de Métricas por Campaña:")
for campaign, metric in metrics.items():
    print(f"\n📌 Campaña: {campaign}")
    print("🔹 Entrenamiento:")
    for key, value in metric["Train"].items():
        print(f"   {key}: {value:.4f}")
    print("🔸 Prueba:")
    for key, value in metric["Test"].items():
        print(f"   {key}: {value:.4f}")

# Crear lista para guardar coeficientes por campaña
all_coefficients = []

# Iterar el modelo, extraer los coefficientes y agregar a la lista
for campaign, model in models.items():
    coefficients = regressor_coefficients(model)
    coefficients['campaign'] = campaign  # Add a campaign column
    all_coefficients.append(coefficients)

# Generar un unico dataframe de coeficientes
all_coefficients_df = pd.concat(all_coefficients)

# Crear una tabla para mostrar los resultados hallados
comparison_table = all_coefficients_df.pivot(index='regressor', columns='campaign', values='coef').round(7)
print("\nTabla de Comparación de Coeficientes (obtenidos del entrenamiento):")
print(comparison_table)

# Crear grilla de gráficos de coeficientes
n_campaigns = len(models)
cols = 3  # Number of columns in the grid
rows = (n_campaigns + cols - 1) // cols  

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)
axes = axes.flatten()  

# Graficar los coeficientes para cada campaña
for i, campaign in enumerate(models.keys()):
    ax = axes[i]
    df_plot = all_coefficients_df[all_coefficients_df['campaign'] == campaign]
    df_plot.plot(x='regressor', y='coef', kind='bar', ax=ax)
    ax.set_title(f'Campaña: {campaign}')
    ax.set_xlabel("Regresor")
    ax.set_ylabel("Coeficiente") 

# Elminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------#

# Definición de horizonte, períodos e inicio para usar la evaluación de metricas de bondad de ajuste con validación cruzada
campaigns = df['campaign'].unique()

cv_results = {}

for campaign in campaigns:
    print(f"Performing cross-validation for campaign: {campaign}")

    df_campaign = df[df['campaign'] == campaign].copy()

    initial = min(365, int(len(df_campaign) * 0.8))  # Use 60% of data for initial training or 365 days, whichever is smaller
    horizon = '90 days'
    period = '30 days'

    try:
        df_cv = cross_validation(models[campaign],  
                                 horizon=horizon,
                                 period=period,
                                 initial=f'{initial} days')
        cv_results[campaign] = df_cv  

        print(f"\nCross-validation results for campaign: {campaign}")
        print(df_cv.head())

    except ValueError as e:
        print(f"Error during cross-validation for campaign {campaign}: {e}")
        print("Consider adjusting initial, horizon, or period values for this campaign.")

# campaign_cv_results = cv_results['Campaign Name']

# --------------------------------------------------------------------------------------------------------------#

# Utilizar la definición de validación cruzada evaluar la métrica MAPE (se cita a modo de ejemplo, se aplicó en todas las métricas de bondad de ajuste)

campaigns = list(cv_results.keys())  

n_campaigns = len(campaigns)
cols = 3
rows = (n_campaigns + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)
axes = axes.flatten() 

for i, campaign in enumerate(campaigns):
    ax = axes[i]  # Get the current subplot
    df_cv = cv_results[campaign]  
    
    df_p = performance_metrics(df_cv)
    
    plot_cross_validation_metric(df_cv, metric='mape', ax=ax)  
    ax.set_title(f"Cross-validation MAPE for Campaign: {campaign}") 

# Elminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------#

# Graficar la componente "Estacionalidad"
component_to_plot = 'weekly'

campaigns_with_models = [c for c in campaigns if c in models]
n_campaigns_valid = len(campaigns_with_models)

if n_campaigns_valid > 0:
        cols = 3  
    rows = (n_campaigns_valid + cols - 1) // cols  

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False) 
    axes = axes.flatten()  
    fig.suptitle(f"Componente: Estacionalidad {component_to_plot.capitalize()} por Campaña (Entrenamiento)", fontsize=16, y=1.02)

    i = 0 # Counter for valid campaigns
    for campaign in campaigns_with_models:
        ax = axes[i]
        model = models[campaign]

             df_campaign = df[df['campaign'] == campaign].copy()
        train_size = int(len(df_campaign) * 0.8)
        train_df = df_campaign.iloc[:train_size].copy()

        required_cols_for_predict = ['ds'] + [col for col in covariables if col in train_df.columns]
        # Filter required_cols_for_predict to only include columns actually in train_df
        required_cols_for_predict = [col for col in required_cols_for_predict if col in train_df.columns]

        try:
            forecast_train = model.predict(train_df[required_cols_for_predict])

            if component_to_plot in forecast_train.columns:
                plot_forecast_component(model, forecast_train, component_to_plot, ax=ax)
                ax.set_title(f"{campaign}")
                ax.set_xlabel("Fecha") # Explicitly label the x-axis
                ax.grid(True, linestyle="--", linewidth=0.5)
            else:
                 ax.set_title(f"{campaign} - No encontrado")
                 ax.text(0.5, 0.5, f'{component_to_plot} component not in forecast', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='orange')

        except Exception as e:
            ax.set_title(f"{campaign} - Error")
            ax.text(0.5, 0.5, f'Plotting error: {e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            print(f"❌ Error al generar forecast o graficar la estacionalidad semanal para la campaña '{campaign}' en entrenamiento: {e}")


        i += 1

# Elminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

else:
    print("\nNo hay campañas con modelos entrenados disponibles para visualizar la estacionalidad semanal en una grilla.")

print("\n✅ Visualización del componente de estacionalidad semanal (Entrenamiento) en grilla completada.")

# --------------------------------------------------------------------------------------------------------------#

# Graficar la componente "Tendencia"
component_to_plot = 'trend'

campaigns_with_models = [c for c in campaigns if c in models]
n_campaigns_valid = len(campaigns_with_models)

if n_campaigns_valid > 0:
    cols = 3  
    rows = (n_campaigns_valid + cols - 1) // cols  

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)
    axes = axes.flatten()  
    fig.suptitle(f"Componente: {component_to_plot.capitalize()} por Campaña (Entrenamiento)", fontsize=16, y=1.02)

    i = 0 # Counter for valid campaigns
    for campaign in campaigns_with_models:
        ax = axes[i]
        model = models[campaign]

        df_campaign = df[df['campaign'] == campaign].copy()
        train_size = int(len(df_campaign) * 0.8)
        train_df = df_campaign.iloc[:train_size].copy()

        required_cols_for_predict = ['ds'] + [col for col in covariables if col in train_df.columns]

        if not all(col in train_df.columns for col in required_cols_for_predict):
             ax.set_title(f"{campaign} - Datos faltantes")
             ax.text(0.5, 0.5, 'Missing regressor columns', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='orange')
             print(f"⚠️ Advertencia: Columnas de regresores faltantes en train_df para la campaña '{campaign}'. No se puede graficar la tendencia.")
             i += 1 # Increment counter even if skipping plot
             continue

        try:
            forecast_train = model.predict(train_df[required_cols_for_predict])

            if component_to_plot in forecast_train.columns:
                plot_forecast_component(model, forecast_train, component_to_plot, ax=ax)
                ax.set_title(f"{campaign}")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle="--", linewidth=0.5)
                ax.set_xlabel("Fecha")
            else:
                 ax.set_title(f"{campaign} - No encontrado")
                 ax.text(0.5, 0.5, f'{component_to_plot} component not in forecast', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='orange')

        except Exception as e:
            ax.set_title(f"{campaign} - Error")
            ax.text(0.5, 0.5, f'Plotting error: {e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            print(f"❌ Error al generar forecast o graficar la tendencia para la campaña '{campaign}' en entrenamiento: {e}")

        i += 1 

# Elminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

else:
    print("\nNo hay campañas con modelos entrenados disponibles para visualizar la tendencia en una grilla.")

print("\n✅ Visualización del componente de tendencia (Entrenamiento) en grilla completada.")

# --------------------------------------------------------------------------------------------------------------#

# Graficar la componente "Holidays"
component_to_plot = 'holidays'

campaigns_with_models = [c for c in campaigns if c in models]
n_campaigns_valid = len(campaigns_with_models)

if n_campaigns_valid > 0:
    cols = 3 
    rows = (n_campaigns_valid + cols - 1) // cols  

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False) 
    axes = axes.flatten()  
    fig.suptitle(f"Componente: {component_to_plot.capitalize()} por Campaña (Entrenamiento)", fontsize=16, y=1.02)

    i = 0 # Counter for valid campaigns
    for campaign in campaigns_with_models:
        ax = axes[i]
        model = models[campaign]

        df_campaign = df[df['campaign'] == campaign].copy()
        train_size = int(len(df_campaign) * 0.8)
        train_df = df_campaign.iloc[:train_size].copy()

        required_cols_for_predict = ['ds'] + [col for col in covariables if col in train_df.columns]
        required_cols_for_predict = [col for col in required_cols_for_predict if col in train_df.columns]

        try:
            forecast_train = model.predict(train_df[required_cols_for_predict])

            if component_to_plot in forecast_train.columns:
                plot_forecast_component(model, forecast_train, component_to_plot, ax=ax)
                ax.set_title(f"{campaign}")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle="--", linewidth=0.5)
                # Set the x-axis label to "Fecha"
                ax.set_xlabel("Fecha")
            else:
                 ax.set_title(f"{campaign} - No encontrado")
                 ax.text(0.5, 0.5, f'{component_to_plot} component not in forecast', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='orange')

        except Exception as e:
            ax.set_title(f"{campaign} - Error")
            ax.text(0.5, 0.5, f'Plotting error: {e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            print(f"❌ Error al generar forecast o graficar los días festivos para la campaña '{campaign}' en entrenamiento: {e}")

        i += 1 

# Elminar gráficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

else:
    print("\nNo hay campañas con modelos entrenados disponibles para visualizar los días festivos en una grilla.")

print("\n✅ Visualización del componente de días festivos (Entrenamiento) en grilla completada.")

# --------------------------------------------------------------------------------------------------------------#

# Graficar los valores observados y los predichos
all_predictions_df = pd.concat(all_predictions)

n_campaigns = len(campaigns)
cols = 3  
rows = (n_campaigns + cols - 1) // cols  

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)
axes = axes.flatten()  # Aplanar ejes para iterar fácilmente

for i, campaign in enumerate(campaigns):
    ax = axes[i]
    df_plot = all_predictions_df[all_predictions_df['campaign'] == campaign]
    ax.plot(df_plot['ds'], df_plot['y'], label='Real (y)', color='green')
    ax.plot(df_plot['ds'], df_plot['yhat'], label='Predicción (yhat)', color='lightblue')
    ax.set_title(f'Campaña: {campaign}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Valor')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

# Eliminar graficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------#

# Graficar los valores observados y los predichos separando el conjunto de datos por train/test
all_predictions_df = pd.concat(all_predictions)

n_campaigns = len(campaigns)
cols = 3
rows = (n_campaigns + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False)
axes = axes.flatten()

for i, campaign in enumerate(campaigns):
    ax = axes[i]

    df_campaign = df[df['campaign'] == campaign].copy()
    train_size = int(len(df_campaign) * 0.8)
    train_df = df_campaign.iloc[:train_size]
    test_df = df_campaign.iloc[train_size:]
    pred_df = all_predictions_df[all_predictions_df['campaign'] == campaign]

    split_date = test_df.iloc[0]['ds']

    ax.plot(train_df['ds'], train_df['y'], label='Train (y)', color='gray')
    ax.plot(test_df['ds'], test_df['y'], label='Test (y)', color='green')
    ax.plot(pred_df['ds'], pred_df['yhat'], label='Predicción (yhat)', color='lightblue')

    ax.axvline(pd.to_datetime(split_date), color='red', linestyle='--', label='Inicio Test')

    ax.set_title(f'Campaña: {campaign}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Valor')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

# Eliminar graficos vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustar la esctructura y mostrar los gráficos
plt.tight_layout()
plt.show()
