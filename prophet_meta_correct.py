#####################################
#FACEBOOK PROPHET APPROACH UNIVARIATE   WERKT!!
#####################################
import pandas as pd
import prophet.plot
import tabulate
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
from prophet.plot import plot_plotly, plot_components_plotly
#import plotly as plt
#insert data that comes from the fed api!! in csv format

meta4 = pd.read_csv('inflation_exp.csv')


meta2 = meta4.rename(columns={"Date": "ds", "inflation expectation %": "y"})#.set_index('ds')
meta2.drop(meta2.tail(0).index,inplace=True)
#meta1 = meta2.drop(['Open', "High", 'Adj Close', 'Low', 'Volume'], axis=1)
#print(meta1)
print(meta2.dtypes)
#print(tabulate(meta1, headers='keys', tablefmt='psql'))
#meta1['ds'] = meta1.to_datetime(meta1['ds'])
meta3 = meta2.dropna()

print(meta3)
model = Prophet()
fit = model.fit(meta3)
future = model.make_future_dataframe(periods=60, freq='M')
future.tail()
#print(tabulate(future, headers='keys', tablefmt='psql'))
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
#print(tabulate(forecast, headers='keys', tablefmt='psql'))
plot_plotly(model, forecast)
plot_components_plotly(model, forecast)
# plt.show()
prophet.plot.plt.show()
# plt.show(fig1)
# plt.show(fig2)
