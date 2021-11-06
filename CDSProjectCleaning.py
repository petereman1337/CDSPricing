# -*- coding: utf-8 -*-
"""
Cleaning

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# importing data

Tickers = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='Tickers')
Tsy = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='Tsy')
CPI = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='CPI')
IndPro = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='IndPro')
dGDP = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='dGDP')
VIX = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='VIX')
MostLiquidBonds = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='MostLiquidBonds')
BondPrices = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='BondPrices')
StockPrices = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='StockPrices')
CDSPrices = pd.read_excel(r'C:\Users\Rexo2\Documents\HYGDataset.xlsx', sheet_name='CDSPrices')

# cleaning CDSPrices

CDSPrices_clean = CDSPrices.iloc[1:,1::2].astype(float).fillna(method='bfill').fillna(method='ffill').dropna(axis=1, how='all')
CDSPrices_clean = CDSPrices_clean.reindex(sorted(CDSPrices_clean.columns), axis=1)
dates = CDSPrices.iloc[1:,0]
CDSPrices_clean = pd.concat([dates,CDSPrices_clean],axis=1)
CDSPrices_clean = CDSPrices_clean.drop(['F.1','NLSN.1','RIG.1','UAL.1','BPYU','CVC'], axis=1)

cds_tickers = CDSPrices_clean.columns[1:].astype(list)
cds_tickers_unique = []
for i in range(len(cds_tickers)):
    if '.' not in cds_tickers[i]:
        cds_tickers_unique.append(cds_tickers[i])
        
cds_tickers = pd.DataFrame(cds_tickers)
cds_tickers.columns = ['ticker']
rec_rates = Tickers.loc[:,['ticker','recovery rate']].drop_duplicates()
rec_rates = pd.merge(rec_rates,cds_tickers, how = 'inner', on = 'ticker')

rec_rates.iloc[106,1] = .15
rec_rates.iloc[107,1] = .15
rec_rates = rec_rates.drop_duplicates()

# compute probs of default
avgdefaultprobs = []
defaultprobs = []
for i in range(len(CDSPrices_clean.T)-1):
    defaultprob = 1 - np.exp((-1*(CDSPrices_clean.iloc[:,i+1]/100))/(1-rec_rates.iloc[i,1]))
    defaultprobs.append(defaultprob)
    avgdefaultprobs.append(np.mean(defaultprob))

defaultprobs_df = pd.DataFrame(defaultprobs).T
defaultprobs = np.asarray(defaultprobs)
defaultprobs_flat = defaultprobs.flatten()

plt.hist(defaultprobs_flat,bins = 10, density = False)
plt.xlabel('Implied Probability of Default')
plt.ylabel('Frequency')
plt.show()

avgdefaultprobs = pd.concat([cds_tickers,pd.DataFrame(avgdefaultprobs)],axis=1)
avgdefaultprobs.columns = ['ticker','avgdefaultprob']



# cleaning StockPrices

StockPrices_clean = StockPrices.iloc[1:,1::2].astype(float).fillna(method='bfill').fillna(method='ffill').dropna(axis=1, how='all')
dates = StockPrices.iloc[1:,0]
StockPrices_clean = pd.concat([dates,StockPrices_clean],axis=1)

Returns_clean = []
for i in range(len(StockPrices_clean)-1):
    x = StockPrices_clean.iloc[:,1:]
    Return_clean = (x.iloc[i+1,:] / x.iloc[i,:])-1
    Returns_clean.append(Return_clean)
Returns_clean = pd.DataFrame(Returns_clean)

stock_tickers = Returns_clean.columns.astype(list)
print(stock_tickers)
        
cds_no_stock = []
for i in range(len(cds_tickers_unique)):
    if cds_tickers_unique[i] not in stock_tickers:
        cds_no_stock.append(cds_tickers_unique[i])

for i in range(len(cds_no_stock)):
    Returns_clean[cds_no_stock[i]] = 0
    StockPrices_clean[cds_no_stock[i]] = 0

StockPrices_clean = StockPrices_clean.reindex(sorted(StockPrices_clean.columns), axis=1)
Returns_clean = Returns_clean.reindex(sorted(Returns_clean.columns), axis=1)
    

# cleaning BondPrices (not using yet)

BondPrices_clean = BondPrices.iloc[:,1::2]
dates = StockPrices.iloc[1:,0]
BondPrices_clean = pd.concat([dates,BondPrices_clean],axis=1)



# This function just computes the mean squared error

def MSE(y, pred):
    return np.round((sum((y - pred)**2))/len(y),3)


# This function plots the main diagonal;for a "predicted vs true" plot with perfect predictions, all data lies on this line

def plotDiagonal(xmin, xmax):
    xsamples = np.arange(xmin,xmax,step=0.01)
    plt.plot(xsamples,xsamples,c='black')

# This helper function plots x vs y and labels the axes

def plotdata(x=None,y=None,xname=None,yname=None,margin=0.05,plotDiag=True,zeromin=False):
    plt.scatter(x,y,label='data')
    plt.xlabel(xname)
    plt.ylabel(yname)
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    if plotDiag:
        plotDiagonal(min(x)-margin*range_x,max(x)+margin*range_x)
    if zeromin:
        plt.xlim(.6,.9)
        plt.ylim(.2,1)
    else:
        plt.xlim(min(x)-margin*range_x,max(x)+margin*range_x)
        plt.ylim(min(y)-margin*range_y,max(y)+margin*range_y)
    plt.show()

# This function plots the predicted labels vs the actual labels (We only plot the first 1000 points to avoid slow plots)

def plot_pred_true(test_pred=None, test_y=None, max_points = 1000):
    plotdata(test_pred[1:max_points], test_y[1:max_points],'Predicted Prob. of Default', 'True Prob. of Default', zeromin=True)
    
# This function runs OLS and bypasses any SVD (Singular Value Decomposition) convergence errors by refitting the model

def run_OLS(train_y, test_y, train_vals, test_vals):
    ols_model = sm.regression.linear_model.OLS(train_y, train_vals)
    while True: # Bypasses SVD convergence assertion error
        try:
            results = ols_model.fit()
            break
        except:
            None
            
    w = np.array(results.params).reshape([len(results.params),1])

    train_pred = np.matmul(train_vals,w)
    test_pred = np.matmul(test_vals,w)

    train_MSE = MSE(train_y, train_pred.flatten())
    test_MSE = MSE(test_y, test_pred.flatten())
      
    print(w)
    return train_MSE, test_MSE, test_pred


# Regress stock price to prob of default

# Train/test split
np.random.seed(69)
stockvsdefaultprob_ols = pd.concat([StockPrices_clean.iloc[:,:-1],defaultprobs_df],axis=1)

stockvsdefaultprob_ols = stockvsdefaultprob_ols.sample(frac=1)
n = len(stockvsdefaultprob_ols)
train_proportion = 0.8
t = int(train_proportion*n)

train_x = np.asarray(stockvsdefaultprob_ols.iloc[0:t,0:153].reset_index(drop=True)).flatten()
test_x = np.asarray(stockvsdefaultprob_ols.iloc[t:,0:153].reset_index(drop=True)).flatten()
train_y = np.asarray(stockvsdefaultprob_ols.iloc[0:t,153:].reset_index(drop=True)).flatten()
test_y = np.asarray(stockvsdefaultprob_ols.iloc[t:,153:].reset_index(drop=True)).flatten()   

train_offset = np.ones(len(train_x)).reshape(len(train_x),1)
test_offset = np.ones(len(test_x)).reshape(len(test_x),1)

train_vals = np.concatenate((train_x.reshape(len(train_x),1),train_offset),axis=1)
test_vals = np.concatenate((test_x.reshape(len(test_x),1),test_offset),axis=1)

train_MSE, test_MSE, test_pred = run_OLS(train_y, test_y, train_vals, test_vals)
plot_pred_true(test_pred.flatten(), test_y)

print("Train MSE\t", str(train_MSE))
print("Test MSE\t", str(test_MSE))
yhat = test_pred

corr1 = np.round(np.cov(train_x,train_y)[0,1] / (np.std(train_x)*np.std(train_y)),3)
print(corr1)

resid = yhat[:,0]-test_y
resid2 = resid[1:]
resid1 = resid[:-1]

def dw(e2,e1):
    sumn = 0
    sumd = 0
    for i in range(len(e2)):
        sumni = (e2[i] - e1[i])**2
        sumdi = e2[i]**2
        sumn += sumni
        sumd += sumdi
    return sumn/sumd

dw(resid2,resid1)

resid3 = np.asarray(np.asarray(StockPrices_clean.iloc[:,:-1].T).flatten())
resid5 = resid3[1:]
resid4 = resid3[:-1]

dw(resid5,resid4)

# Clean MostLiquidBonds
MLB = MostLiquidBonds.loc[:,['ticker','issue date','rating','sector']]
OLB = MLB.sort_values(by = 'issue date', ascending = False).drop_duplicates(subset='ticker')
OLB_clean = OLB.sort_values(by = 'ticker', ascending = True).reset_index(drop=True)
OLB_clean['avgdefaultprob'] = avgdefaultprobs.iloc[:,1]
OLB_clean = pd.DataFrame(OLB_clean)

table1 = OLB_clean['avgdefaultprob'].groupby(OLB_clean['rating']).describe()
table2 = OLB_clean['avgdefaultprob'].groupby(OLB_clean['sector']).describe()


# Regress stock price and rating (manyhot) to prob of default
ratingtypes = np.asarray(OLB_clean.loc[:,'rating'].explode().unique())
print(len(ratingtypes))

Ratings_series = StockPrices_clean.iloc[:,:-1]
for i in range(len(Ratings_series.T)):
    Ratings_series.iloc[:,i] = OLB_clean.loc[i,'rating']

def manyhot(point,vector):
    v = np.zeros(len(vector))
    for i in range(len(v)):
        if point == vector[i]:
            v[i] = 1
    return v

for i in range(len(Ratings_series)):
    for j in range(len(Ratings_series.T)):
        Ratings_series.iloc[i,j] = manyhot(Ratings_series.iloc[i,j],ratingtypes)


Ratings_series_random = Ratings_series.reindex_like(stockvsdefaultprob_ols)
train_ratings = Ratings_series_random.iloc[0:t,0:153].reset_index(drop=True)
test_ratings = Ratings_series_random.iloc[t:,0:153].reset_index(drop=True) 

train_ratings_flat = pd.DataFrame(list(np.asarray(train_ratings.T).flatten()))
test_ratings_flat = pd.DataFrame(list(np.asarray(test_ratings.T).flatten()))

train_vals2 = np.concatenate((train_x.reshape(len(train_x),1),train_ratings_flat,train_offset),axis=1)
test_vals2 = np.concatenate((test_x.reshape(len(test_x),1),test_ratings_flat,test_offset),axis=1)

train_MSE2, test_MSE2, test_pred2 = run_OLS(train_y, test_y, train_vals2, test_vals2)
plot_pred_true(test_pred2.flatten(), test_y)

print("Train MSE\t", str(train_MSE2))
print("Test MSE\t", str(test_MSE2))
yhat2 = test_pred2

# Regress stock price, rating, sector (manyhot) to prob of default

