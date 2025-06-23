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


def phi_fn(feature):
    '''
    Auxiliary function used in the conformal prediction, 
    indicating the type of the item sold in each auction
    '''
    scalar_values = np.array(feature[:,dimension_x:(dimension_x+dimension_z)])
    # Initialize the indicator matrix
    matrix = np.zeros((len(np.array(feature[:,1])), number_z))
    # Fill in the indicator matrix
    for i, value in enumerate(scalar_values):
        for j in range(0, number_z):
            if  np.array_equal(value, z_domain[j]):
                matrix[i, j] = 1

    return matrix