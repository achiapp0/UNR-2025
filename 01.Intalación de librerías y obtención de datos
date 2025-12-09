# Instalar Modelo Prophet
!pip install prophet

# Instalar Librería para Detección de Outliers
!pip install pyod==1.1.0

# Importar librerías necesarias
import io
import os
import json
import pickle
import joblib
import datetime
import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error,  mean_absolute_percentage_error
from pyod.models.copod import COPOD
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
from scipy.stats import wilcoxon
from statsmodels.robust import mad
from prophet.diagnostics import cross_validation
from prophet.utilities import regressor_coefficients
from prophet.plot import plot_forecast_component
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation

import warnings
warnings.filterwarnings("ignore")

#Conectar Colab con Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cargar el conjunto de datos originales
df = pd.read_parquet('/content/drive/MyDrive/Prophet/data.parquet')
