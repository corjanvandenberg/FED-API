import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

start = datetime(2022, 9, 1)
end = datetime(2023, 1, 5)

XXX = pdr.DataReader('^DJI', 'stooq') #pijltje omhoog moet erbij voor symbol
print(XXX)
print(XXX.shape)

#aex = pd.read_csv("C:/Users\cjmin\PycharmProjects\pythonProject444\h_world_txt\data\hourly\world\indices/aex.txt")
#print(aex)