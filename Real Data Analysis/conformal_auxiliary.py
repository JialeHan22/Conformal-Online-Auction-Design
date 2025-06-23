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
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from conditionalconformal import CondConf
import random
from scipy.spatial import ConvexHull



def phi_fn(feature):
    '''
    Auxiliary function used in the conformal prediction, 
    indicating the type of the item sold in each auction
    '''
    scalar_values = np.array(feature[:,3])

    # Initialize the indicator matrix
    matrix = np.zeros((len(scalar_values), 3))
    value_z = np.array(["syschannel","michael-33","saveking"])

    # Fill in the indicator matrix
    for i, value in enumerate(scalar_values):
        for j in range(0,3):
            if  value == value_z[j]:
                matrix[i, j] = 1

    return matrix