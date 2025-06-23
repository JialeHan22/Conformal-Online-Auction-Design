import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, DMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
from conditionalconformal import CondConf
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull


# import functions and values from other modules 
from conformal_auxiliary import phi_fn
from Myerson_auction import myerson_reserve


alpha = 0.1
def conformal_predict(m=1000, z_0=3, N=1000):
    '''
    For a new item, calculate the revenue of different mechanisms.

    Args:
        m: number of bidders
        z_0: new item's feature
        N: number of data points

    Returns:
        payment: revenue of the COAD
        myerson_value: revenue of the Empirical Myerson Auction
        myerson_reserve_price: the single reserve price of the Empirical Myerson Auction
        sb_value: revenue of the second-price auction
        max_welfare: maximum social welfare
    '''

    
    # Generate historical data
    Z_space = [3, 5, 7]
    z = np.random.choice(Z_space, N)
    x_sd = 1  
    epsilon_x = np.random.normal(loc=z/10, scale=x_sd, size=N)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = 0.2 * x[t-1] + np.sqrt(1 - 0.2**2) * epsilon_x[t]
    y = np.exp(x) * z + ((x**2)*np.cos(z))*(truncnorm.rvs(-1, 1, loc=0, scale=1, size=N))
    
    ############################################################################################ 
    ''' The Empirical Myerson Auction '''
    D_myerson = pd.DataFrame({'x': x, 'y': y, 'z': z})
    # Extract the historical auction data corresponding to the current new item from past datasets
    df_myerson =  D_myerson[D_myerson['z'] == z_0]
    myerson_reserve_price = myerson_reserve(df_myerson)
    ############################################################################################ 

    ''' COAD implementation '''
    # Split the data into D_cali and D_train
    half_N = int(N/2)
    D_cali = pd.DataFrame({'x': x[:half_N], 'y': y[:half_N], 'z': z[:half_N]})
    D_train = pd.DataFrame({'x': x[half_N:], 'y': y[half_N:], 'z': z[half_N:]})
    
    # fit a eighth order polynomial
    poly = PolynomialFeatures(8)
    X_poly = poly.fit_transform(D_train[['x', 'z']])
    reg = LinearRegression().fit(X_poly, D_train['y'])    

    # Conformal prediction setup
    score_fn = lambda feature,  y : abs(y - reg.predict(poly.transform(pd.DataFrame({'x': feature[:,0], 'z': feature[:,1]}))))
    condCovProgram = CondConf(score_fn, phi_fn, {})
    condCovProgram.setup_problem(D_cali[['x', 'z']].to_numpy(),D_cali['y'].to_numpy())

    # Generate data for the new auction
    z_new = np.full(m, z_0)
    epsilon_x_new = np.random.normal(loc=z_new/10, scale=x_sd, size=m)
    x_new = np.zeros(m)
    for t in range(1, m):
        x_new[t] = 0.2 * x_new[t-1] + np.sqrt(1 - 0.2**2) * epsilon_x_new[t]
    y_new = np.exp(x_new) * z_new + ((x_new **2)*np.cos(z_new))*(truncnorm.rvs(-1, 1, loc=0, scale=1, size=m))
    new_data = pd.DataFrame({'x': x_new, 'y': y_new, 'z': z_new})
    dtest_new = pd.DataFrame(new_data[['x', 'z']])
    predicted_values_poly_new = reg.predict(poly.transform(dtest_new))

    # Calculate the half-length of the prediction interval
    Xtest = np.array([new_data[['x', 'z']].to_numpy()[3,:]])
    d = condCovProgram.predict(1-alpha, Xtest, lambda x, y : x)

    # Calculate the bidder-specific reserve prices 
    reserve_price = predicted_values_poly_new - d
    # Calculate the pseudo-virtual values
    virtual_value = new_data['y'] * (new_data['y'] > reserve_price)
    
    # Find the index(es) of the maximum virtual value
    max_virtual_value_indexes = np.where(virtual_value == np.max(virtual_value))[0]
    
    # Break the tie in the winning sets:
    winner = None
    if len(max_virtual_value_indexes) > 1:
    # Check the reserve price for the max virtual value indexes
        max_reserve_price_indexes = np.where(reserve_price[max_virtual_value_indexes] == np.max(reserve_price[max_virtual_value_indexes]))[0]
        if len(max_reserve_price_indexes) > 1:
        # If multiple maximums, choose one at random
            winner = np.random.choice(max_virtual_value_indexes[max_reserve_price_indexes])
        else:
        # If only one maximum
            winner = max_virtual_value_indexes[max_reserve_price_indexes[0]]
    else:
    # If only one maximum virtual value
        winner = max_virtual_value_indexes[0]
    
    ############################################################################################ 

    ''' Revenue Calculation '''
    # Calculate the revenue of the COAD
    sorted_virtual_value = np.sort(virtual_value)[::-1]
    payment = max(0, sorted_virtual_value[1], reserve_price[winner])

    # Calculate the maximum social welfare
    max_welfare = np.max(new_data['y'])

    # Calculate the revenue of the second-price auction
    sb_value = np.sort(new_data['y'])[::-1][1] if len(np.sort(new_data['y'])[::-1]) >= 2 else 0

    # Calculate the revenue of the Empirical Myerson Auction
    myerson_value = np.maximum(np.sort(new_data['y'])[::-1][1], myerson_reserve_price) if np.sort(new_data['y'])[::-1][0] >= myerson_reserve_price else 0

    return payment, myerson_value, myerson_reserve_price, sb_value, max_welfare