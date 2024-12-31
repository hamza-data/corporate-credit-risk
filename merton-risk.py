import yfinance as yf 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import nasdaqdatalink as ndl
import math
from scipy.stats import norm 
from scipy.optimize import minimize
import fundamentalanalysis as fa
api_key = 'd12ec091e9acd7181564ff6cdf83cada'
ndl.ApiConfig.api_key = 'GsX3qg2VYPinD1YHLgmF'


###############################################################
    
ticker = 'AAPL'
sigma_a0ini = 0.1
mult_lower_a0ini = 0.001 
mult_upper_a0ini = 10 
lower_sigma_a = 0.001
upper_sigma_a = 4

balance_sheet = fa.balance_sheet_statement(ticker, api_key, period="annual")
enterprise = fa.enterprise(ticker, api_key, period="annual")
interest = ndl.get("USTREASURY/YIELD").loc[:,'1 YR']/100
prices = pd.DataFrame(yf.download(ticker)['Adj Close'])
prices['ds'] = prices.index

 
end_date = datetime.strptime(balance_sheet.loc['fillingDate'][0], '%Y-%m-%d').date()
start_date = end_date - timedelta(365)
end_date = str(end_date)
start_date = str(start_date)
sliced_prices = prices.loc[(prices['ds'] >= start_date) & (prices['ds'] <= end_date)] 
sliced_prices = sliced_prices.drop('ds', axis = 1)
ret = np.log(sliced_prices).diff(periods = 1).iloc[1:]
sigma_e = ret.std() * math.sqrt(252)

payables = balance_sheet.loc['accountPayables',][0]
current_debt = np.nan_to_num(balance_sheet.loc['shortTermDebt',][0], nan = 0)
short_term_debt = payables + current_debt

long_term_debt = balance_sheet.loc['longTermDebt',][0]

d = short_term_debt + 0.5 * long_term_debt

try:
    r = interest.loc[end_date]
except:
    end_date = datetime.strptime(balance_sheet.loc['fillingDate'][0], '%Y-%m-%d').date()
    end_date = end_date - timedelta(1)
    end_date = str(end_date)
    r = interest.loc[end_date]

n = enterprise.loc['numberOfShares'].iloc[0]
p0 = float(sliced_prices.iloc[-1])
e0 = n * p0

t = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days / 365

a0ini = e0 + d

def func(par, e0, sigma_e, r, t, d):
    a0 = par[0]
    sigma_a = par[1]
    d1 = (math.log(a0/d) + (r + sigma_a ** 2/ 2) * t)/(sigma_a * math.sqrt(t))
    d2 = d1 - sigma_a * math.sqrt(t)
    return ((e0 - a0 * norm.cdf(d1) + math.exp(-r * t) * d * norm.cdf(d2)) ** 2 + (sigma_e * e0 - norm.cdf(d1) * sigma_a * a0) ** 2)                  

bounds = ((a0ini * mult_lower_a0ini, a0ini * mult_upper_a0ini),(lower_sigma_a, upper_sigma_a))
result = minimize(fun = func, x0 = (a0ini, sigma_a0ini), method = "L-BFGS-B", args = (e0, sigma_e, r, t, d), bounds = bounds)

a0 = result.x[0]
sigma_a = result.x[1]
d1 = (math.log(a0/d) + (r + sigma_a ** 2/ 2) * t)/(sigma_a * math.sqrt(t))
d2 = d1 - sigma_a * math.sqrt(t)    
prob = norm.cdf(-d2)

print(d2)
print(prob)




