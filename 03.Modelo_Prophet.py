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

# Before initializing the Prophet model, inspect the holidays_df
print("Inspecting holidays_df:")
print(holidays_df.head())
print(holidays_df['holiday'].dtype)

# If the 'holiday' column is not of object/string type, convert it
if holidays_df['holiday'].dtype != 'object':
    print("Converting 'holiday' column to string type.")
    holidays_df['holiday'] = holidays_df['holiday'].astype(str)
    print("Conversion complete. New dtype:", holidays_df['holiday'].dtype)

# Now proceed with the rest of the code that was failing:
# Lista de covariables a incluir en el modelo - OJO REVISAR PREVIOUS CYBER DAYS!!!!
covariables = ['clicks', 'impressions', 'visitors', 'visits', 'orders', 'total_units_net',
               'budget_USD', 'cost_USD']

# Lista de campañas únicas
campaigns = df['campaign'].unique()

# Diccionarios para modelos, predicciones y métricas
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

    # Agregar regresores
    for cov in covariables:
        model.add_regressor(cov)

    # Entrenar modelo
    model.fit(train_df[['ds', 'y'] + covariables])
    models[campaign] = model

    # Crear DataFrame futuro
    future = test_df[['ds'] + covariables].copy()
    forecast = model.predict(future)
    forecasts[campaign] = forecast

    # Agregar valores reales
    forecast['campaign'] = campaign
    forecast['y'] = test_df['y'].values
    all_predictions.append(forecast[['ds', 'y', 'yhat', 'campaign']])

    # Calcular métricas
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
