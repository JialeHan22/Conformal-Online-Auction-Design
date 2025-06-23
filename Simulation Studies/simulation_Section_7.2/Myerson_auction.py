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


from data_generation import z_domain, function_1, dimension_x, dimension_z, number_z, prepare_data_for_regression
from conformal_auxiliary import phi_fn

'''
Implement the empirical Myerson auction, following the method of Cole and Roughgarden (2014)
'''

def create_coordinates(v):
    ''' Construct the "empirical quantile" '''
    m = len(v)
    coordinates = np.zeros((m, 2))  

    for j in range(m):
        coordinates[j, 0] = (2 * j + 1) / (2 * m) 
        coordinates[j, 1] = (2 * j + 1) * v[j] / (2 * m)  

    return coordinates

def get_slope(x, hull_points, slopes):
    ''' Calculate the empirical ironed virtual value for each value '''
    for i in range(len(hull_points) - 1):
        x1, x2 = hull_points[i][0], hull_points[i + 1][0]
        if x1 <= x <= x2:
            if x == x2 and i < len(slopes) - 1:
                return max(slopes[i], slopes[i + 1])
            if x == x1 and i > 0:
                return max(slopes[i - 1], slopes[i])
            return slopes[i]
    return None


def find_last_min_non_negative_index(nums):
    ''' Find the index of the reserve price '''
    min_non_neg = float('inf')  
    min_index = -1 
    for i, num in enumerate(nums):
        if num >= 0 and num <= min_non_neg:
            min_non_neg = num
            min_index = i

    return min_index



def myerson_reserve(table):
    '''
    Calculates a single reserve price using historical data, treating all bidders as symmetric. 
    
    Args:  
        data_table (pd.DataFrame): Historical bidding data containing a values vector "T_1".  
        
    Returns:  
        float: Computed reserve price.  
    '''
    sorted_bids = np.sort(table["T_1"])[::-1]  
    
    new_vector_1 = create_coordinates(sorted_bids)
    new_points = np.array([[0, 0], [1, 0]])
    new_vector_1 = np.concatenate((new_points[0:1], new_vector_1, new_points[1:2]), axis=0)
    points = new_vector_1
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    hull_points = hull_points[hull_points[:, 0].argsort()]

    slopes = []
    for i in range(len(hull_points) - 1):
        x1, y1 = hull_points[i]
        x2, y2 = hull_points[i + 1]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

    point_slopes = []
    for point in points:
        x = point[0]
        slope = get_slope(x, hull_points, slopes)
        point_slopes.append(slope)
        
    index_reserve = find_last_min_non_negative_index(point_slopes[1:-1])
    return sorted_bids[index_reserve]
