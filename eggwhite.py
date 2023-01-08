import pandas as pd
from FED_api import *   #import all functions from FED_api.py file
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from sklearn import preprocessing
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


begin_date = '1950-01-01'
end_date = datetime.datetime.now()
df_datetime = pd.DataFrame({'Date': pd.date_range(start=begin_date,end=end_date, freq='D')})

def dried_egg():
    df_datetime_chickenstuff = pd.DataFrame({'Date': pd.date_range(start='2005-01-01', end=end_date, freq='D')})
    merge2 = pd.merge(df_datetime_chickenstuff, eggs(), how='inner', on='Date')
    merge2['calc dried whole eggs (usd/kg)'] = merge2['Eggs, grade A large cost per dozen (in USD)']*12.6
    #merge2['egg 10sma'] = merge2['Eggs, grade A large cost per dozen (in USD)'].rolling(2).mean() #10 SMA op de egg prices
    print(tabulate(merge2, headers = 'keys', tablefmt = 'psql'))
    plt.plot(merge2['Date'], merge2['calc dried whole eggs (usd/kg)'], color='g', label='calc dried whole eggs (usd/kg)')
    plt.legend(loc='best')
    plt.show()

def egg_prediction_pd():
    global merge4
    global merge5
    df_datetime_egg = pd.DataFrame({'Date': pd.date_range(start='2010-01-01', end=end_date, freq='D')})
    merge = pd.merge(df_datetime_egg, WTI(), how='outer', on='Date') # deze als eerste gedaan omdat dit de grootste dataset is
    merge2 = pd.merge(merge, cash_assets_all_comm_banks(), how='outer', on='Date')
    merge3 = pd.merge(merge2, corn(), how='outer', on='Date')
    merge4 = pd.merge(merge3, eggs(), how='right', on='Date') #inner omdat alleen de eierprijzen relevant zijn
    #print(tabulate(merge4, headers='keys', tablefmt='psql'))
    #print(merge.head())
    merge5 = merge4.dropna(thresh=3)
    #print(tabulate(merge5, headers='keys', tablefmt='psql'))
    #print(merge5.tail())
    # fig, ax1 = plt.subplots(figsize=(8, 8))
    # ax2 = ax1.twinx()
    # ax2.plot(merge3['Date'], merge3['Eggs, grade A large cost per dozen (in USD)'], color='g', label='Eggs, grade A large cost per dozen (in USD)')
    # #ax1.plot(merge4['Date'], merge4['soybean meal (usd/metric ton)'], color='r', label='soybean meal (usd/metric ton)')
    # ax1.plot(merge3['Date'], merge3['corn  (usd/metric ton)'], color='lightgreen', label='corn  (usd/metric ton)')
    # ax1.plot(merge3['Date'], merge3['crude oil price WTI, cushing okl (usd per barrel)'], color='purple', label='crude oil price WTI, cushing okl (usd per barrel)')
    # ax1.legend(loc='best')
    # ax2.legend(loc='lower right')
    #merge4.to_csv('eggs.csv')
    # plt.show()

egg_prediction_pd()


#alle data interpoleren!
print(merge5.to_string())
merge5['crude oil price WTI, cushing okl (usd per barrel)'] = merge5['crude oil price WTI, cushing okl (usd per barrel)'].interpolate()
merge5['corn  (usd/metric ton)'] = merge5['corn  (usd/metric ton)'].interpolate()
merge5['cash assets all commercial banks (usd)'] = merge5['cash assets all commercial banks (usd)'].interpolate()
merge5['Eggs, grade A large cost per dozen (in USD)'] = merge5['Eggs, grade A large cost per dozen (in USD)'].interpolate()
print(merge5.to_string())

# merge5.plot(x='Date')
# plt.show()

#selecteer de X en Y waardes. Note values!!
#X = merge4[['corn  (usd/metric ton)','crude oil price WTI, cushing okl (usd per barrel)']].values
#Y = merge4[['Eggs, grade A large cost per dozen (in USD)']].values
#print(tabulate(Y, headers='keys', tablefmt='psql'))


# SOURCE OF THIS SCRIPT: https://unit8co.github.io/darts/examples/01-multi-time-series-and-covariates.html
import numpy as np
import torch
import matplotlib.pyplot as plt

from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    StatsForecastAutoARIMA,
    NBEATSModel,
    BlockRNNModel,
    XGBModel,
    VARIMA,
)

from darts import TimeSeries
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
#from darts.models.forecasting import sf_auto_arima
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset, ElectricityDataset


#multi_serie_elec = merge4
# X = merge4[['corn  (usd/metric ton)','crude oil price WTI, cushing okl (usd per barrel)']].values
# Y = merge4[['Eggs, grade A large cost per dozen (in USD)']].values
multi_serie_elec = TimeSeries.from_dataframe(merge5, time_col='Date', freq='MS', fill_missing_dates=False) #MS = month start.  https://stackoverflow.com/questions/35339139/what-values-are-valid-in-pandas-freq-tags
# retaining only three components in different ranges
retained_components = ['cash assets all commercial banks (usd)','corn  (usd/metric ton)','crude oil price WTI, cushing okl (usd per barrel)','Eggs, grade A large cost per dozen (in USD)']
multi_serie_elec = multi_serie_elec[retained_components]

# multi_serie_elec.plot()
# multi_serie_elec.plot()
# plt.show()


# split in train/validation sets
training_set, validation_set = multi_serie_elec[:-30], multi_serie_elec[-30:]
# define a scaler, by default, normalize each component between 0 and 1
scaler_dataset = Scaler()
# scaler is fit on training set only to avoid leakage
training_scaled = scaler_dataset.fit_transform(training_set)
validation_scaled = scaler_dataset.transform(validation_set)

# split in train/validation sets
def fit_and_pred(model, training, validation):
    model.fit(training)
    forecast = model.predict(len(validation*2))
    return forecast

#input variable for NN models
#model_VARIMA = VARIMA(p=6, d=0, q=0, trend="n") #p = aantal maanden
model_XGboost = XGBModel(lags=12) #p = aantal maanden
#model_autoVARIMA = StatsForecastAutoARIMA(season_length=12) # only supports univariate?
model_nBeats = NBEATSModel(input_chunk_length = 12, output_chunk_length = 24)

#input variable RNN model
# model_GRU = RNNModel(
#     input_chunk_length=12,
#     model="LSTM",
#     hidden_dim=25,
#     n_rnn_layers=2,
#     training_length=36,
#     n_epochs=200,
# )

# training and prediction with the VARIMA model
#forecast_VARIMA = fit_and_pred(model_VARIMA, training_scaled, validation_scaled)
#print("MAE (VARIMA) = {:.2f}".format(mae(validation_scaled, forecast_VARIMA)))

# forecast_autoVARIMA = fit_and_pred(model_autoVARIMA, training_scaled, validation_scaled)
# print("MAE (autoVARIMA) = {:.2f}".format(mae(validation_scaled, forecast_autoVARIMA)))

forecast_XGboost = fit_and_pred(model_XGboost, training_scaled, validation_scaled)
print("MAE (model_XGboost) = {:.2f}".format(mae(validation_scaled, forecast_XGboost)))

future = model_XGboost.predict(n = 50)
futureX = scaler_dataset.inverse_transform(future)
plt.plot(futureX)
plt.show()
# forecast_Nbeats = fit_and_pred(model_nBeats, training_scaled, validation_scaled)
#print("MAE (model_XGboost) = {:.2f}".format(mae(validation_scaled, forecast_Nbeats)))

#training and prediction with the RNN model
# forecast_RNN = fit_and_pred(model_GRU, training_scaled, validation_scaled)
# print("MAE (RNN) = {:.2f}".format(mae(validation_scaled, forecast_RNN)))

#forecast_VARIMA = scaler_dataset.inverse_transform(forecast_VARIMA)
forecast_XGboost = scaler_dataset.inverse_transform(forecast_XGboost)
#forecast_Nbeats = scaler_dataset.inverse_transform(forecast_Nbeats)
#forecast_VARIMA = scaler_dataset.inverse_transform(forecast_autoVARIMA)
#forecast_RNN = scaler_dataset.inverse_transform(forecast_RNN)

labels = [f"forecast {component}" for component in retained_components]
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
# validation_set.plot(ax=axs[0])
# forecast_Nbeats.plot(label=labels, ax=axs[0])
# axs[0].set_ylim(0, 5)
# axs[0].set_title("Nbeats model forecast")
# axs[0].legend(loc="upper left")
# validation_set.plot(ax=axs[1])
# forecast_XGboost.plot(label=labels, ax=axs[1])
# axs[1].set_ylim(0, 5)
# axs[1].set_title("xgboost model forecast")
# axs[1].legend(loc="upper left")
# plt.show()


#####################################
#FACEBOOK PROPHET APPROACH UNIVARIATE   WERKT!!
#####################################
#'Date' must be renamed to 'ds' and 'eggs must be renamed to 'y'. Also needs CSV file. UNIVARIATE!!
# meta1 = merge4.rename(columns={"Date": "ds", "Eggs, grade A large cost per dozen (in USD)": "y"}).set_index('ds')
# meta2 = meta1.drop(columns=['crude oil price WTI, cushing okl (usd per barrel)','corn  (usd/metric ton)'])
# meta3 = meta2.to_csv('meta.csv')
# meta4 = pd.read_csv('meta.csv')
# #meta3 =
# print(tabulate(meta4, headers='keys', tablefmt='psql'))
# print(meta4.head())
# model = Prophet()
# fit = model.fit(meta4)
# future = model.make_future_dataframe(periods=12, freq='M')
# future.tail()
# #print(tabulate(future, headers='keys', tablefmt='psql'))
# forecast = model.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# fig1 = model.plot(forecast)
# fig2 = model.plot_components(forecast)
#print(tabulate(forecast, headers='keys', tablefmt='psql'))
#plot_plotly(model, forecast)
# plot_components_plotly(model, forecast)
# plt.show()
# plt.show(fig1)
# plt.show(fig2)



#CALCULATIONS/ASSUMPTIONS BELOW!!!!!
# cost_dozen_eggs = 3.5 #usd
# egg_cost = cost_dozen_eggs/12
# h20_perc_large_egg = 85 #%
# total_weight_large_egg = 63.75 #gr
# edible_weight_large_egg = 44 #gr source: https://www.pizzamaking.com/forum/index.php?topic=2563.0
# total_dm_egg = edible_weight_large_egg*((100-h20_perc_large_egg)/100) #gr
# total_weight_dozen_eggs = total_dm_egg*12 #gr
# total_h20_evaporated_per_dozen = (edible_weight_large_egg*12)*(100/h20_perc_large_egg) #gr
# energy_needed_for_1L_evap = 336 #kJ  source: https://www.quora.com/How-do-I-calculate-the-energy-required-to-evaporate-water
# combust_1L_of_gas_energy_release = 55500 #kJ
# energy_needed_to_dry_dozen_eggs = total_h20_evaporated_per_dozen*energy_needed_for_1L_evap/1000 #kJ/dozen eggs
# liters_gas_needed_for_dozen_eggs = energy_needed_to_dry_dozen_eggs/combust_1L_of_gas_energy_release
# gas_price = 1 #dollar/liter
# evaporation_needed_for_1kg_dried_egg = 1*(1/((100-h20_perc_large_egg)/100))
# energy_needed_for_kg_dried_egg = energy_needed_for_1L_evap*evaporation_needed_for_1kg_dried_egg
# gas_cost_kg_eggs = energy_needed_for_kg_dried_egg/combust_1L_of_gas_energy_release*gas_price
# eggs_needed_1kg_dried_eggs = 1000/(44*((100-h20_perc_large_egg)/100))
# cost_kg_dried_eggs = eggs_needed_1kg_dried_eggs*egg_cost + gas_cost_kg_eggs
#print(eggs_needed_1kg_dried_eggs/12)
#print(cost_kg_dried_eggs)


