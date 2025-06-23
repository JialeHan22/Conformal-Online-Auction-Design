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


alpha = 0.1
def conformal_predict_z_0(alpha=0.1, m=2500, z_0=3):
    '''
    For a new item, simulate an auction and calculate the coverage probability of the conformal prediction interval over 1000 experiments.

    Args:
        alpha: miscoverage rate
        m: number of bidders
        z_0: new item's feature


    Returns:
        IR: vectors of the coverage rate
        d: half-length of the interval in the last experiment
    '''
    IR = []
    for j in range(1000):
        Z_space = [3, 5, 7]
        N = 1000
        # Generate historical data
        z = np.random.choice(Z_space, N)
        x_sd = 1  
        epsilon_x = np.random.normal(loc=z/10, scale=x_sd, size=N)
        x = np.zeros(N)
        for t in range(1, N):
            x[t] = 0.2 * x[t-1] + np.sqrt(1 - 0.2**2) * epsilon_x[t]
        y = np.exp(x) * z + ((x**2)*np.cos(z))*(truncnorm.rvs(-1, 1, loc=0, scale=1, size=N) )
        # Split the data into D_cali and D_train
        half_N = int(N/2)
        D_cali = pd.DataFrame({'x': x[:half_N], 'y': y[:half_N], 'z': z[:half_N]})
        D_train = pd.DataFrame({'x': x[half_N:], 'y': y[half_N:], 'z': z[half_N:]})
        
        # Fit a eighth order polynomial
        poly = PolynomialFeatures(8)
        X_poly = poly.fit_transform(D_train[['x', 'z']])
        reg = LinearRegression().fit(X_poly, D_train['y'])
        score_fn = lambda feature,  y : abs(y - reg.predict(poly.transform(pd.DataFrame({'x': feature[:,0], 'z': feature[:,1]}))))
        condCovProgram = CondConf(score_fn, phi_fn, {})
        condCovProgram.setup_problem(D_cali[['x', 'z']].to_numpy(),D_cali['y'].to_numpy())

        # Generate the new data
        z_new = np.full(m, z_0)
        epsilon_x_new = np.random.normal(loc=z_new/10, scale=x_sd, size=m)
        x_new = np.zeros(m)
        for t in range(1, m):
            x_new[t] = 0.2 * x_new[t-1] + np.sqrt(1 - 0.2**2) * epsilon_x_new[t]
        y_new = np.exp(x_new) * z_new + ((x_new **2)*np.cos(z_new))*(truncnorm.rvs(-1, 1, loc=0, scale=1, size=m))
        new_data = pd.DataFrame({'x': x_new, 'y': y_new, 'z': z_new})
        dtest_new = pd.DataFrame({'x': x_new, 'z': z_new})
        predicted_values_poly_new = reg.predict(poly.transform(dtest_new))

        # Conformal prediction setup
        Xtest = np.array([new_data[['x', 'z']].to_numpy()[11,:]])
        d = condCovProgram.predict(1-alpha, Xtest, lambda x, y : x)
        
        # Calculate bounds of intervals
        hat_t_L = predicted_values_poly_new - d
        hat_t_U = predicted_values_poly_new + d
        
        # Calculate the converage probability
        IR.append(np.mean((hat_t_L <= new_data['y']) & (new_data['y'] <= hat_t_U) ))
    return IR, d