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

'''
Construct the regression function μ(x, z) and the provide the data generation function used to generate the simulated data
'''

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

dimension_z = 20
dimension_x = 20
number_z = 30
# Generating 30 vectors, each with 20 dimensions
z_domain = np.random.normal(size=(number_z, dimension_z))

# Generating beta_1 and beta_2
beta_1 = np.random.uniform(-0.5, 0.5, dimension_x)
beta_2 = np.random.uniform(-0.5, 0.5, dimension_z)


def generate_function(beta_1, beta_2):
    '''Data generation function'''

    def function_2(X, Z):
        epsilon = np.random.uniform(-1, 1)

        T = exp(np.dot(beta_1.T, X))*np.dot(beta_2.T, Z) + exp(np.cos(np.dot(X, Z))**2) * (epsilon)
        return T


    return function_2


function_2 = generate_function(beta_1, beta_2)