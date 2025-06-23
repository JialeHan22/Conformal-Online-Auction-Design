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


'''
Randomly split the historical data into training data and calibration data
'''


def find_remaining_numbers(original_numbers, selected_numbers):
    remaining_numbers = [num for num in original_numbers if num not in selected_numbers]
    return remaining_numbers


def split(Table):
    Table = Table.reset_index(drop=True)
    train_df = pd.DataFrame(columns=Table.columns)
    cali_df = pd.DataFrame(columns=Table.columns)
    N = len(Table)
    half_N = int(N/2)
    selected_numbers = random.sample(list(range(1, N + 1)), half_N)
    train_N = sorted(selected_numbers)
    cali_N = find_remaining_numbers(list(range(1, N + 1)),train_N)
    cali_N = [x - 1 for x in cali_N ]
    train_N = [x - 1 for x in train_N]
    train_df = Table.loc[train_N]
    cali_df = Table.loc[cali_N]

    return train_df, cali_df