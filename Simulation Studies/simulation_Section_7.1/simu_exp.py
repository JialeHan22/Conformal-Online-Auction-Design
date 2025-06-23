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
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
import conditionalconformal
from conditionalconformal import CondConf
import itertools
import math
from numpy import sin, cos, exp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

# import functions from different modules
from data_generation import z_domain, function_2, dimension_x, dimension_z, number_z
from conformal_auxiliary import phi_fn
from coverage import flatten_features


np.random.seed(1)
Z_new = z_domain[np.random.randint(z_domain.shape[0])]
var = 1
def conformal_predict(alpha, m, Z_new, myerson_reserve_price, model_nn, condCovProgram):
    '''
    For a new item, calculate the revenue of different mechanisms.

    Args:
        alpha: miscoverage rate
        m: number of bidders
        Z_new: new item's feature
        myerson_reserve_price: the single reserve price of the Empirical Myerson Auction
        model_nn: fitted neural network
        condCovProgram: conformal prediction model 

    Returns:
        payment: revenue of the COAD
        myerson_value: revenue of the Empirical Myerson Auction
        sb_value: revenue of the second-price auction
        max_welfare: maximum social welfare
    '''
    
    ''' Data generation for a new auction '''
    data_points_new = []
    for _ in range(m):
        # Compute the mean of the square of Z
        mean_x_new = np.mean(Z_new**2)
        # Generate a 20-dimensional vector X from a normal distribution with mean mean_x_new and variance 1
        X_new = np.random.normal(mean_x_new, var, dimension_x)
        # Compute values using the generated functions
        T_1_new = function_2(X_new, Z_new)
        data_points_new.append([X_new, Z_new, T_1_new])
        
    D_new = pd.DataFrame(data_points_new, columns=["X", "Z", "T_1"])

    
     #####################################################

    ''' COAD implementation '''
    X_neww = flatten_features(D_new)
    T_1_new= D_new['T_1']
    T_1_pred_new = model_nn.predict(X_neww, verbose=0).flatten()
    Xtest = np.array([flatten_features(D_new)[1,:]])
    # Calculate the half-length of the conformal prediction interval
    d = condCovProgram.predict(1-alpha, Xtest, lambda x, y : x)

    # Get the bidder-specific reserve prices
    reserve_price = T_1_pred_new - d
    # Calculate the pseudo-virtual values
    virtual_value = np.array(T_1_new) * (np.array(T_1_new) >= reserve_price)

    # Find the index(es) of the maximum virtual value, which is also the winning sets
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

    #####################################################

    ''' Revenue Calculation For Different Mechanisms '''
    # Calculate the revenue of the COAD
    sorted_virtual_value = np.sort(virtual_value)[::-1]
    payment = max(0, sorted_virtual_value[1], reserve_price[winner])

    # Calculate the maximum social welfare
    max_welfare = np.max(np.array(T_1_new))

    # Calculate the revenue of the second-price auction
    sb_value = np.sort(np.array(T_1_new))[::-1][1] if len(np.sort(np.array(T_1_new))[::-1]) >= 2 else 0

    # Calculate the revenue of the Empirical Myerson Auction
    myerson_value = np.maximum(np.sort(np.array(T_1_new))[::-1][1], myerson_reserve_price) if np.sort(np.array(T_1_new))[::-1][0] >= myerson_reserve_price else 0

    
    return payment, myerson_value, sb_value, max_welfare