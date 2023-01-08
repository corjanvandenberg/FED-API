# linkjes die nuttig zijn
# achtergrond info: https://github.com/mortada/fredapi
# https://mortada.net/python-api-for-fred.html
# belangrijkste fed data, descending https://fred.stlouisfed.org/tags/series?t=bea%3Bid&rt=id&ob=pv&od=desc

from fredapi import Fred
import pandas as pd
import openpyxl
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

key = "e33c7caa6cc3694f9985cdc1ca3ecd6e"
fred = Fred(api_key=key)
###################
#DATA OPVRAGEN!!!!
###################

#series van bepaalde tijdsvak importeren!
# s = fred.get_series('SP500', observation_start='2014-09-02', observation_end='2014-09-05')
#print(s.tail())


# haal een series op van de FED site. de code is the vinden naast titel vn grafiek: https://fred.stlouisfed.org/series/ATNHPIUS14260Q
# belangrijke IDs: PCE (personal consumption expenditures M), GFDEGDQ188S (Federal debt: total public debt as % of GPD 3M),

#######################
## ECONOMIC INDICATORS
#######################

def inflation_expectation():
    global df_MICH
    df_MICH = pd.DataFrame(fred.get_series("MICH"))
    df_MICH.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_MICH.columns = ['Date', 'inflation expectation %']
    df_MICH.to_csv('inflation_exp.csv', index=False)
    #print(df_PMSAVE)
    return df_MICH


def sp500():
    global df_SP500
    df_SP500 = pd.DataFrame(fred.get_series("SP500"))
    df_SP500.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_SP500.columns = ['Date', 'S&P 500']
    #print(df_PMSAVE)
    return df_SP500

def DOW():
    global df_DJIA
    df_DJIA = pd.DataFrame(fred.get_series("DJIA"))
    df_DJIA.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_DJIA.columns = ['Date', 'DOW Jones Index']
    #print(df_PMSAVE)
    return df_DJIA

def Wilshire5000():
    global df_WILL5000INDFC
    df_WILL5000INDFC = pd.DataFrame(fred.get_series("WILL5000INDFC"))
    df_WILL5000INDFC.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_WILL5000INDFC.columns = ['Date', 'Wilshire 5000 total market cap index']
    #print(df_PMSAVE)
    return df_WILL5000INDFC

def bitcoin():
    global df_CBBTCUSD
    df_CBBTCUSD = pd.DataFrame(fred.get_series("CBBTCUSD"))
    df_CBBTCUSD.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_CBBTCUSD.columns = ['Date', 'bitcoin usd']
    #print(df_PMSAVE)
    df_CBBTCUSD.to_csv('bitcoin2.csv', index = False)
    return df_CBBTCUSD


def ethereum():
    global df_CBETHUSD
    df_CBETHUSD = pd.DataFrame(fred.get_series("CBETHUSD"))
    df_CBETHUSD.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_CBETHUSD.columns = ['Date', 'Ethereum usd']
    #print(df_PMSAVE)
    df_CBETHUSD.to_csv('ethereum.csv', index=False)
    return df_CBETHUSD

#######################
#VARIOUS INTEREST RATES
#######################
def interest_on_reserve_balances():
    global df_IORB
    df_IORB = pd.DataFrame(fred.get_series("IORB"))
    df_IORB.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_IORB.columns = ['Date', 'interest on reserve balances fed (%)']
    #print(df_PMSAVE)
    return df_IORB

def SOFR_30MA():
    global df_SOFR30DAYAVG
    df_SOFR30DAYAVG = pd.DataFrame(fred.get_series("SOFR30DAYAVG"))
    df_SOFR30DAYAVG.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_SOFR30DAYAVG.columns = ['Date', 'secured overnight financing round 30d avg (%)']
    #print(df_PMSAVE)
    return df_SOFR30DAYAVG

def fed_fund_target_rate_upper_limit():
    global df_DFEDTARU
    df_DFEDTARU = pd.DataFrame(fred.get_series("DFEDTARU"))
    df_DFEDTARU.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_DFEDTARU.columns = ['Date', 'fed target rate, upper limit (%)']
    #print(df_PMSAVE)
    return df_DFEDTARU

def fed_funding_rate():
    global df_FEDFUNDS
    df_FEDFUNDS = pd.DataFrame(fred.get_series("FEDFUNDS"))
    df_FEDFUNDS.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_FEDFUNDS.columns = ['Date', 'federal bank funding rate (%)']
    #print(df_PMSAVE)
    return df_FEDFUNDS

def bank_prime_loan_rate():
    global df_DPRIME
    df_DPRIME = pd.DataFrame(fred.get_series("DPRIME"))
    df_DPRIME.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_DPRIME.columns = ['Date', 'bank prime loan rate (%)']
    #print(df_DPRIME)
    return df_DPRIME

def mortgage_rate_15y_avg_usa():
    global df_MORTGAGE15US
    df_MORTGAGE15US = pd.DataFrame(fred.get_series("MORTGAGE15US"))
    df_MORTGAGE15US.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_MORTGAGE15US.columns = ['Date', '15y mortgage rate avg in usa (%)']
    #print(df_PMSAVE)
    return df_MORTGAGE15US

def mortgage_rate_30y_avg_usa():
    global df_MORTGAGE30US
    df_MORTGAGE30US = pd.DataFrame(fred.get_series("MORTGAGE30US"))
    df_MORTGAGE30US.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_MORTGAGE30US.columns = ['Date', '30y mortgage rate avg in usa (%)']
    #print(df_PMSAVE)
    return df_MORTGAGE30US

#######################
#POPULATION / WORKFORCE
#######################

def unemployment_rate():
    global df_U6RATE
    df_U6RATE = pd.DataFrame(fred.get_series("U6RATE"))
    df_U6RATE.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_U6RATE.columns = ['Date', 'unemployment rate usa (%)']
    #print(df_PMSAVE)
    return df_U6RATE

def unemployment_rate_min_job_losers():
    global df_U2RATE
    df_U2RATE = pd.DataFrame(fred.get_series("U2RATE"))
    df_U2RATE.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_U2RATE.columns = ['Date', 'unemployment rate ==> job losers usa (%)']
    #print(df_PMSAVE)
    return df_U2RATE

def population_usa():
    global df_POPTHM
    df_POPTHM = pd.DataFrame(fred.get_series("POPTHM"))#, columns=['date', 'pi'])
    df_POPTHM.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_POPTHM.columns = ['Date', 'population usa']
    #print(df_PMSAVE)
    return df_POPTHM

def interest_payments_usa_gov():
    global df_A091RC1Q027SBEA
    df_A091RC1Q027SBEA = pd.DataFrame(fred.get_series("A091RC1Q027SBEA"))#, columns=['date', 'pi'])
    df_A091RC1Q027SBEA.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_A091RC1Q027SBEA.columns = ['Date', 'interest payments of usa gov (in USD)']
    #print(df_PMSAVE)
    return df_A091RC1Q027SBEA

######################
#CONSUMER DATA FINANCIALS
########################
def consumer_loans_usa():
    global df_CCLACBW027SBOG
    df_CCLACBW027SBOG = pd.DataFrame(fred.get_series("CCLACBW027SBOG"))#, columns=['date', 'pi'])
    df_CCLACBW027SBOG.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_CCLACBW027SBOG.columns = ['Date', 'consumer loans usa (in USD)']
    #print(df_PMSAVE)
    return df_CCLACBW027SBOG

def personal_savings():
    global df_PMSAVE
    df_PMSAVE = pd.DataFrame(fred.get_series("PMSAVE"))#, columns=['date', 'pi'])
    df_PMSAVE.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PMSAVE.columns = ['Date', 'personal savings (in USD)']
    #print(df_PMSAVE)
    return df_PMSAVE

def personal_consumption_excl_food_energy():
    global df_PCEPILFE
    df_PCEPILFE = pd.DataFrame(fred.get_series("PCEPILFE"))#, columns=['date', 'pi'])
    df_PCEPILFE.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PCEPILFE.columns = ['Date', 'Pers. consumption excl food/energy (in USD)']
    #print(df_PCEPILFE)
    return df_PCEPILFE

def personal_income():
    global df_pi
    df_pi = pd.DataFrame(fred.get_series("PI"))#, columns=['date', 'pi'])
    df_pi.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_pi.columns = ['Date', 'PI (in USD)']
    #print(type(df_pii))
    return df_pi

def real_disposable_income():
    global df_realpi
    df_realpi = pd.DataFrame(fred.get_series("DSPIC96"))#, columns=['date', 'pi'])
    df_realpi.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_realpi.columns = ['Date', 'real disposable income (in USD)']
    #print(type(df_realpii))
    return df_realpi

def personal_savings_rate():
    global df_PSAVERT
    df_PSAVERT = pd.DataFrame(fred.get_series("PSAVERT"))
    df_PSAVERT.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PSAVERT.columns = ['Date', 'personal savings rate (%)']
    #print(df_PMSAVE)
    return df_PSAVERT


#######################
#BANKS ASSETS##########
#######################
def cash_assets_all_comm_banks():
    global df_CASACBW027SBOG
    df_CASACBW027SBOG = pd.DataFrame(fred.get_series("CASACBW027SBOG"))
    df_CASACBW027SBOG.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_CASACBW027SBOG.columns = ['Date', 'cash assets all commercial banks (usd)']
    #print(df_CASACBW027SBOG)
    return df_CASACBW027SBOG

#cash_assets_all_comm_banks()

def total_assets_fed_usa():
    global df_WALCL
    df_WALCL = pd.DataFrame(fred.get_series("WALCL"))#, columns=['date', 'pi'])
    df_WALCL.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_WALCL.columns = ['Date', 'total assets FED, wednesday level (in usd)']
    #print(df_WALCL)
    return df_WALCL

total_assets_fed_usa()

def overnight_rev_repo():
    global df_RRPONTSYD
    df_RRPONTSYD = pd.DataFrame(fred.get_series("RRPONTSYD"))#, columns=['date', 'pi'])
    df_RRPONTSYD.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_RRPONTSYD.columns = ['Date', 'overnight rev repo. treasury sold by fed (in Bn usd)']
    #print(df_PMSAVE)
    return df_RRPONTSYD

#############
## FOOD / COMMODITIES
#############

def eggs(): #  Monthly https://fred.stlouisfed.org/series/APU0000708111
    global df_APU0000708111
    df_APU0000708111 = pd.DataFrame(fred.get_series("APU0000708111"))#, columns=['date', 'pi'])
    df_APU0000708111.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_APU0000708111.columns = ['Date', 'Eggs, grade A large cost per dozen (in USD)']
    #print(tabulate(df_APU0000708111, headers='keys', tablefmt='psql'))
    #print(df_PMSAVE)
    df_APU0000708111.to_excel("output.xlsx")
    return df_APU0000708111


def chicken_breast(): #monthly https://fred.stlouisfed.org/series/APU0000FF1101
    global df_APU0000FF1101
    df_APU0000FF1101 = pd.DataFrame(fred.get_series("APU0000FF1101"))  # , columns=['date', 'pi'])
    df_APU0000FF1101.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_APU0000FF1101.columns = ['Date', 'avg chicken breast boneless (cost usd / pound)']
    # print(df_PMSAVE)
    return df_APU0000FF1101

def sunflower(): #monthly https://fred.stlouisfed.org/series/APU0000FF1101
    global df_PSUNOUSDM
    df_PSUNOUSDM = pd.DataFrame(fred.get_series("PSUNOUSDM"))  # , columns=['date', 'pi'])
    df_PSUNOUSDM.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PSUNOUSDM.columns = ['Date', 'sunflower (usd/metric ton)']
    # print(df_PMSAVE)
    return df_PSUNOUSDM

def barley(): #monthly https://fred.stlouisfed.org/series/APU0000FF1101
    global df_PBARLUSDM
    df_PBARLUSDM = pd.DataFrame(fred.get_series("PBARLUSDM"))  # , columns=['date', 'pi'])
    df_PBARLUSDM.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PBARLUSDM.columns = ['Date', 'barley (usd/metric ton)']
    # print(df_PMSAVE)
    return df_PBARLUSDM

def soybean_meal(): #monthly https://fred.stlouisfed.org/series/APU0000FF1101
    global df_PSMEAUSDM
    df_PSMEAUSDM = pd.DataFrame(fred.get_series("PSMEAUSDM"))  # , columns=['date', 'pi'])
    df_PSMEAUSDM.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PSMEAUSDM.columns = ['Date', 'soybean meal (usd/metric ton)']
    # print(df_PMSAVE)
    return df_PSMEAUSDM

def corn(): #monthly https://fred.stlouisfed.org/series/APU0000FF1101
    global df_PMAIZMTUSDM
    df_PMAIZMTUSDM = pd.DataFrame(fred.get_series("PMAIZMTUSDM"))  # , columns=['date', 'pi'])
    df_PMAIZMTUSDM.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PMAIZMTUSDM.columns = ['Date', 'corn  (usd/metric ton)']
    # print(df_PMSAVE)
    return df_PMAIZMTUSDM

def beef():
    global df_PBEEFUSDQ
    df_PBEEFUSDQ = pd.DataFrame(fred.get_series("PBEEFUSDQ"))  # , columns=['date', 'pi'])
    df_PBEEFUSDQ.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PBEEFUSDQ.columns = ['Date', 'beef (usd cents per pound)']
    # print(df_PMSAVE)
    return df_PBEEFUSDQ


def sugar():
    global df_PSUGAISAUSDM
    df_PSUGAISAUSDM = pd.DataFrame(fred.get_series("PSUGAISAUSDM"))  # , columns=['date', 'pi'])
    df_PSUGAISAUSDM.reset_index(inplace=True)  # deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_PSUGAISAUSDM.columns = ['Date', 'sugar (usd cents per pound)']
    # print(df_PMSAVE)
    return df_PSUGAISAUSDM

#####################
# ENERGY############
####################

def WTI(): #  Daily  https://fred.stlouisfed.org/series/DCOILWTICO
    global df_DCOILWTICO
    df_DCOILWTICO = pd.DataFrame(fred.get_series("DCOILWTICO"))#, columns=['date', 'pi'])
    df_DCOILWTICO.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_DCOILWTICO.columns = ['Date', 'crude oil price WTI, cushing okl (usd per barrel)']
    #print(df_PMSAVE)
    return df_DCOILWTICO

def consumer_energy_price(): #  Daily  https://fred.stlouisfed.org/series/DCOILWTICO
    global df_APU000072610
    df_APU000072610 = pd.DataFrame(fred.get_series("APU000072610"))#, columns=['date', 'pi'])
    df_APU000072610.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_APU000072610.columns = ['Date', 'electricity cost avg usa consumer (usd/kwhr)']
    #print(df_PMSAVE)
    return df_APU000072610

def gas_price_regular_USA(): #  Daily  https://fred.stlouisfed.org/series/DCOILWTICO
    global df_GASREGW
    df_GASREGW = pd.DataFrame(fred.get_series("GASREGW"))#, columns=['date', 'pi'])
    df_GASREGW.reset_index(inplace=True) #deze is nodig om de date niet meer index te maken!! Daarnaast voeg je een index row in
    df_GASREGW.columns = ['Date', 'US Regular All Formulations Gas Price (usd/gallon)']
    #print(df_PMSAVE)
    return df_GASREGW



# uitstekende manier om de tabel weer te geven!
#print(tabulate(pi_df, headers = 'keys', tablefmt = 'psql'))
#print(pi_df)


