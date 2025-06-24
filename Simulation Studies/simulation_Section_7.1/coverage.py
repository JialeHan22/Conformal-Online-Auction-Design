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

from data_generation import z_domain, function_2, dimension_x, dimension_z, number_z
from conformal_auxiliary import phi_fn

######################################################################################################################
'''Function to combine X and Z into a single flattened array'''
flatten_features = lambda df: np.array([np.concatenate([x, z]) for x, z in zip(df['X'], df['Z'])])


######################################################################################################################
np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)
Z_new = z_domain[np.random.randint(z_domain.shape[0])]
var_x = 1

def conformal_predict_z_0(alpha=0.1, m=100, Z_new=Z_new, N=1000, t=200):
    '''
    For a new item, calculate the coverage probability of the conformal prediction interval.

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
        data_points = []
        for _ in range(N):
            # Randomly select a vector from z_domain to be Z
            Z = z_domain[np.random.randint(z_domain.shape[0])]
            # Compute the mean of the square of Z
            mean_x = np.mean(Z**2)
            # Generate a 20-dimensional vector X from a normal distribution with mean mean_x and variance 1
            X = np.random.normal(mean_x, var_x, dimension_x)
            # Compute values the generated functions
            T_1 = function_2(X, Z)
            # Append the data point (X, Z, T_1) to the list
            data_points.append([X, Z, T_1])
        # Split the data into D_cali and D_train
        D_cali = pd.DataFrame(data_points[:N//2], columns=["X", "Z", "T_1"])
        D_train = pd.DataFrame(data_points[N//2:], columns=["X", "Z", "T_1"])

        # Preparing data for regression
        X_train = flatten_features(D_train)
        # Extracting T_1 for D_train and D_cali
        Y_train = D_train['T_1'].values
        # Build a neural network model
        model_nn = Sequential([
            InputLayer(input_shape=(X_train.shape[1],)),   
            Dense(128, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.02)), # Use LeakyReLU and L2 regularization
            Dropout(0.3),
            Dense(64, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.02)),
            Dense(1)               
        ])
 
        # Compilation model
        model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model on D train
        history = model_nn.fit(X_train, Y_train, epochs=15, batch_size=32, verbose=0)

        # Conformal prediction setup
        score_fn = lambda feature,  y : abs(y - model_nn.predict(feature, verbose=0).flatten())
        condCovProgram = CondConf(score_fn, phi_fn, {})
        condCovProgram.setup_problem(flatten_features(D_cali),D_cali['T_1'].to_numpy())

        # Generate the new data
        data_points_new = []
        for _ in range(m):
            # Compute the mean of the square of Z
            mean_x_new = np.mean(Z_new**2)
            # Generate a 20-dimensional vector X from a normal distribution with mean mean_x_new and variance 1
            X_new = np.random.normal(mean_x_new, var_x, dimension_x)
            # Compute T_1 using the generated functions
            T_1_new = function_2(X_new, Z_new)
            # Append the data point (X, Z, T_1) to the list
            data_points_new.append([X_new, Z_new, T_1_new])

        D_new = pd.DataFrame(data_points_new, columns=["X", "Z", "T_1"])
        X_neww = flatten_features(D_new)
        T_1_new= D_new['T_1']
        T_1_pred_new = model_nn.predict(X_neww, verbose=0).flatten()

        # Calculate the half-length of the interval
        Xtest = np.array([flatten_features(D_new)[10,:]])
        d = condCovProgram.predict(1-alpha, Xtest, lambda x, y : x)
        
        # Calculate bounds of intervals
        hat_t_L = T_1_pred_new - d
        hat_t_U = T_1_pred_new + d 
        
        # Calculate the converage probability
        IR.append(np.mean((hat_t_L <= T_1_new) & (T_1_new <= hat_t_U)))
             
    return IR, d