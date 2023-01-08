import datetime

import pandas as pd
from FED_api import *   #import all functions from FED_api.py file
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

# print(bank_prime_loan_rate())
# print(mortgage_rate_15y_avg_usa())
#mortgage_premium = pd.DataFrame

##################
#correcte truukjes hieronder hieronder
##################
# simpel lineplot mbv import plotly.express as px
# plot = px.line(df2, title="real disposable inc")
# plot.show()

# correctly refer to a column ==> personal_income()['PI (in USD)']
# insert a column at index 0 to the right
#delta_income['new column'] = real_disposable_income()['real disposable income (in USD)']
# info over merge: how == left betekent dat alle infor van de linker/originele df bewaard moet worden. outer betekent ALLE info van beide dfs

begin_date = '1950-01-01'
end_date = datetime.datetime.now() # '2023-01-10'
df_datetime = pd.DataFrame({'Date': pd.date_range(start=begin_date,end=end_date, freq='D')})
#print((df_datetime))
#print(tabulate(df_datetime, headers = 'keys', tablefmt = 'psql'))

def interest_rates_comp():
    df_datetime_int_rate = pd.DataFrame({'Date': pd.date_range(start='1990-01-01', end=end_date, freq='D')})
    merge = pd.merge(df_datetime_int_rate, fed_funding_rate(), how='inner', on='Date')
    merge2 = pd.merge(merge, bank_prime_loan_rate(), how='inner', on='Date')
    merge3 = pd.merge(merge2, mortgage_rate_15y_avg_usa(),how='outer', on='Date')
    plt.plot(merge3['Date'], merge3['federal bank funding rate (%)'], color='g', label='fed fund rate')
    plt.plot(merge3['Date'], merge3['bank prime loan rate (%)'], color='r', label='bank prime loan rate')
    plt.plot(merge3['Date'], merge3['15y mortgage rate avg in usa (%)'], color='b', label='15y mortgage rate avg in usa (%)')
    plt.legend(loc='best')
    #plt.xlim([datetime.date(2010, 1, 1), datetime.date(2022, 12,29)])  # werkt correct mbt tijd!
    print(tabulate(merge3, headers = 'keys', tablefmt = 'psql'))
    plt.show()

#interest_rates_comp()

def chicken_stuff():
    df_datetime_chickenstuff = pd.DataFrame({'Date': pd.date_range(start='2005-01-01', end=end_date, freq='D')})
    merge = pd.merge(df_datetime_chickenstuff, eggs(), how='inner', on='Date')
    merge2 = pd.merge(merge, chicken_breast(), how='outer', on='Date')
    merge2['egg 5sma'] = merge2['Eggs, grade A large cost per dozen (in USD)'].rolling(5).mean() #10 SMA op de egg prices
    print(tabulate(merge2, headers = 'keys', tablefmt = 'psql'))
    plt.plot(merge2['Date'], merge2['Eggs, grade A large cost per dozen (in USD)'], color='g', label='Eggs, grade A large cost per dozen (in USD)')
    plt.plot(merge2['Date'], merge2['avg chicken breast boneless (cost usd / pound)'], color='r', label='avg chicken breast boneless (cost usd / pound)')
    plt.plot(merge2['Date'], merge2['egg 5sma'], color='lightgreen', label='egg 5 sma')
    plt.legend(loc='best')
    plt.show()

chicken_stuff()



#2 lijntjes in 1 plot
# df3 = pd.concat([personal_income(), real_disposable_income()])
# plt.plot(df3['Date'], df3['PI (in USD)'], color='g', label='PI')
# plt.plot(df3['Date'], df3['real disposable income (in USD)'], color='r', label='realPI')
# plt.show()

#real_disposable_income()
#print(real_disposable_income())
#print(personal_income(pi_df))