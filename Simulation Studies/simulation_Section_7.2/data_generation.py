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

'''
Construct the regression function μ(x, z) and the data generation function.
'''

np.random.seed(1)
random.seed(1)

dimension_z = 100
dimension_x = 100
number_z = 30
# Generating 30 vectors, each with 100 dimensions
z_domain = np.random.normal(size=(number_z, dimension_z))

# Generating beta_1 and beta_2
beta_1 = np.random.uniform(-1, 1, dimension_x)
beta_2 = np.random.uniform(-1, 1, dimension_z)

def generate_function(beta_1, beta_2):
    def function_1(X, Z):
        epsilon = truncnorm.rvs(-1, 1)  # Generate epsilon from a truncated normal distribution
        T = (np.dot(beta_1.T, X) ** 2) * (np.sin(np.dot(beta_2.T, Z)) ** 2) + np.exp(np.cos(np.dot(X, Z))**2) * (epsilon)
        return T
    return function_1

function_1 = generate_function(beta_1, beta_2)


##########################################################################################################
def prepare_data_for_regression(df):
    # Extracting X and Z and converting them into suitable format for regression
    X_data = np.array(df['X'].tolist())
    Z_data = np.array(df['Z'].tolist())

    # Concatenating X and Z
    combined_data = np.concatenate([X_data, Z_data], axis=1)

    return combined_data
