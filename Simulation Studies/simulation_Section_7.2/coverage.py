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
import itertools
import math
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

''' import functions and values from other modules '''
from data_generation import z_domain, prepare_data_for_regression, function_1, dimension_x, dimension_z, number_z
from conformal_auxiliary import phi_fn


np.random.seed(1)
random.seed(1)
Z_new = z_domain[np.random.randint(z_domain.shape[0])]

def conformal_predict_z_0(alpha=0.1, m=100, Z_new=Z_new, N=1000, t=200):
    '''
    For a new item, simulate an auction and calculate the coverage probability of the conformal prediction interval.

    Args:
        alpha: miscoverage rate
        m: number of bidders
        Z_new: new item's feature
        N: numer of data points used in training
        t: the number of experiments


    Returns:
        IR: vectors of the coverage probability
        d: half-length of the interval in the last experiment
    '''
    IR = []
    for j in range(t):
        # Generate historical data
        data_points = []
        for _ in range(N):
            # Randomly select a vector from z_domain to be Z
            Z = z_domain[np.random.randint(z_domain.shape[0])]
            # Compute the mean of the square of Z
            mean_x = np.mean(Z**2)
            # Generate a 100-dimensional vector X from a normal distribution with mean mean_x and variance 1
            X = np.random.normal(mean_x, 1, dimension_x)
            # Compute T_1 using the generated functions
            T_1 = function_1(X, Z)
            # Append the data point (X, Z, T_1) to the list
            data_points.append([X, Z, T_1])
        # Split the data into D_cali and D_train
        D_cali = pd.DataFrame(data_points[:N//2], columns=["X", "Z", "T_1"])
        D_train = pd.DataFrame(data_points[N//2:], columns=["X", "Z", "T_1"])

        # Preparing data for regression
        X_train = prepare_data_for_regression(D_train)
        # Extracting T_1
        T_1_train = D_train['T_1']
        # Using a polynomial model 
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        # Training the model
        model = LinearRegression()
        model.fit(X_train_poly, T_1_train)

        # Conformal prediction setup
        score_fn = lambda feature,  y : abs(y - model.predict(poly.transform(pd.DataFrame(feature))))
        condCovProgram = CondConf(score_fn, phi_fn, {})
        condCovProgram.setup_problem(prepare_data_for_regression(D_cali),D_cali['T_1'].to_numpy())

        # Generate the new data
        data_points_new = []
        for _ in range(m):
            # Compute the mean of the square of Z
            mean_x_new = np.mean(Z_new**2)
            # Generate a 100-dimensional vector X from a normal distribution with mean mean_x and variance 1
            X_new = np.random.normal(mean_x_new, 1, dimension_x)
            # Compute T_1 using the generated functions
            T_1_new = function_1(X_new, Z_new)
            # Append the data point (X, Z, T_1) to the list
            data_points_new.append([X_new, Z_new, T_1_new])

        D_new = pd.DataFrame(data_points_new, columns=["X", "Z", "T_1"])       
        X_neww = prepare_data_for_regression(D_new)
        T_1_new= D_new['T_1']
        X_new_poly = poly.transform(X_neww)
        T_1_pred_new = model.predict(X_new_poly)
        
        # Calculate the half-length of the interval
        Xtest = np.array([prepare_data_for_regression(D_new)[10,:]])
        d = condCovProgram.predict(1-alpha, Xtest, lambda x, y : x) 
        
        # Calculate bounds of intervals
        hat_t_L = T_1_pred_new - d
        hat_t_U = T_1_pred_new + d 
        
        # Calculate the converage probability
        IR.append(np.mean((hat_t_L <= T_1_new) & (T_1_new <= hat_t_U)))
        
            
    return IR, d