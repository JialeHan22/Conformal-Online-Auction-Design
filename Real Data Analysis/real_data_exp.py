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

# import functions from different modules
from split_table import split
from conformal_auxiliary import phi_fn
from Myerson_auction import myerson_reserve


######################################################################################################################
'''
Feature preprocessing pipeline for polynomial regression:
 - Apply 2nd-degree polynomial expansion to numerical features (x1, x2, x3)
 - One-hot encode categorical feature (z)
'''

preprocessor = ColumnTransformer(
    transformers=[
        ('num', PolynomialFeatures(degree=2), ['x1', 'x2', 'x3']),  
        ('cat', OneHotEncoder(), ['z'])
    ])
######################################################################################################################


def delete_none(train_df, new_df):
    '''
    Identifies bidders in the new auction who have previously participated by placing bids
    '''
    unique_bidders = train_df['Bidder'].unique()
    
    new = new_df[new_df['Bidder'].isin(unique_bidders)]

    return new

######################################################################################################################


def single_exp(Table_old,Table_new,N=800,item="syschannel"):
    '''
    For a fixed type of item, calculate the revenue of different mechanisms in each auction using N data points,  
    and the coverage probability of the conformal prediction interval.

    Args:
        Table_old: historical bidding data
        Table_new: corresponding new auction data
        N: numer of data points used in training
        item: item type/feature

    Returns:
        Pay: revenue of the COAD
        new_data_myerson: revenue of the Empirical Myerson Auction
        new_data_2nd: revenue of the second-price auction
        welfare: the maximum social welfare
        IR: coverage probability of the conformal prediction interval
        Myerson_Reserve_Price: the single reserve price of the Empirical Myerson Auction
    '''
    
    IR = []
    Pay = []
    new_data_2nd = []
    welfare = []
    new_data_myerson = []
    Myerson_Reserve_Price = []
    for i in range(0,len(Table_old)):
        df_his = Table_old[i]
        df_his = df_his.sample(n=N)

        ''' The Empirical Myerson Auction '''
        # Get the historical data for the specific item
        df_myerson = df_his[df_his['Seller'] == item]
        # Calculate the single reserve price
        myerson_reserve_price = myerson_reserve(df_myerson)
        Myerson_Reserve_Price.append(myerson_reserve_price)

        #####################################################

        ''' COAD '''
        # Randomly split the historical data into training data and calibration data
        D_train = split(df_his)[0]
        D_cali = split(df_his)[1]
        # Delete bidders who have not previously placed bids in the new auction
        Condition_data = delete_none(Table_old[i], Table_new[i])
        # Clean the data and extract the features
        D_train = pd.DataFrame({'x1': D_train['BidTime'],'x2': D_train['mean_bid'], 'x3': D_train['Bidder Rating'], 'y': D_train['BidAmount'], 'z': D_train['Seller']})
        D_cali = pd.DataFrame({'x1': D_cali['BidTime'],'x2': D_cali['mean_bid'], 'x3': D_cali['Bidder Rating'], 'y': D_cali['BidAmount'], 'z': D_cali['Seller']})
        X = D_train[['x1', 'x2', 'x3', 'z']]
        y = D_train['y']
        # Train the quadratic polynomial regression
        model = make_pipeline(preprocessor, LinearRegression())
        model.fit(X, y)

        # Predict the value of the bidders in the new auction
        D_new = pd.DataFrame({'x1': Condition_data['BidTime'], 'x2': Condition_data['mean_bid'],'x3': Condition_data['Bidder Rating'],'y': Condition_data['BidAmount'], 'z': Condition_data['Seller']})
        X_new = D_new[['x1', 'x2', 'x3','z']]
        y_new = D_new['y']
        new_data = pd.DataFrame({'x1': D_new['x1'],'x2': D_new['x2'], 'x3': D_new['x3'],'z': D_new['z'], 'y': D_new['y']})
        y_new_predict = model.predict(X_new)

        # Apply the conditional conformal prediction and calculate the prediction intervals
        score_fn = lambda feature,  y : abs(y - model.predict(pd.DataFrame({'x1': feature[:,0],'x2': feature[:,1],'x3': feature[:,2], 'z': feature[:,3]})))
        condCovProgram = CondConf(score_fn, phi_fn, {})
        condCovProgram.setup_problem(D_cali[['x1', 'x2', 'x3', 'z']].to_numpy(),D_cali['y'].to_numpy())
        alpha = 0.1
        Xtest = np.array([new_data[['x1', 'x2', 'x3', 'z']].to_numpy()[0,:]])
        d = condCovProgram.predict(1-alpha, Xtest, lambda x, y : x)
        # Lower bound of the prediction interval:
        hat_t_L = y_new_predict - d 
        # Upper bound of the prediction interval:
        hat_t_U = y_new_predict + d 
        
        # Coverage probability of the conformal prediction interval
        IR.append(np.mean((hat_t_L <= new_data['y']) & (new_data['y'] <= hat_t_U)))

        # Get the bidder-specific reserve prices
        reserve_price = hat_t_L
        # Calculate the pseudo-virtual values
        virtual_value = new_data['y'] * (new_data['y'] > reserve_price)

        # Find the index(es) of the maximum virtual value, which is also the winning sets
        max_virtual_value_indexes = np.where(virtual_value == np.max(virtual_value))[0]
        # Break the tie in the winning sets
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

        # Find the second highest pseudo-virtual value (if it exists, otherwise 0)
        sorted_virtual_value = np.sort(virtual_value)[::-1]
        sorted_virtual_value_2 = sorted_virtual_value[1] if len(sorted_virtual_value) >= 2 else 0

        #####################################################

        ''' Revenue Calculation '''
        # Calculate the revenue of the COAD
        payment = max(0, sorted_virtual_value_2, reserve_price[winner],Condition_data['Opening Bid'].iloc[0])
        Pay.append(payment)
        
        # Calculate the maximum social welfare
        max_welfare = np.max(new_data['y'])
        welfare.append(max_welfare)

        # Calculate the revenue of the second-price auction
        new_data_2 = np.sort(new_data['y'])[::-1][1] if len(np.sort(new_data['y'])[::-1]) >= 2 else Condition_data['Opening Bid'].iloc[0]
        new_data_2nd.append(new_data_2)

        # Calculate the revenue of the Empirical Myerson Auction
        new_data_m = np.maximum(np.sort(new_data['y'])[::-1][1], myerson_reserve_price) if np.sort(new_data['y'])[::-1][0] >= myerson_reserve_price else Condition_data['Opening Bid'].iloc[0]
        new_data_myerson.append(new_data_m)
        

    return Pay, new_data_myerson, new_data_2nd, welfare,  IR, Myerson_Reserve_Price

######################################################################################################################


def repeated_exp(Table_old, Table_new, N=700, item="syschannel", num_runs=2):
    '''
    For a fixed type of item, calculate the average revenue of different mechanisms and the coverage probabilities of the 
    conformal prediction intervals in each auction using N data points, based on "num_runs" experiments

    Args:
        Table_old: historical bidding data
        Table_new: corresponding new auction data
        N: numer of data points used in training
        item: item type/feature
        num_runs: number of the experiments

    Returns:
        COAD_matrix: revenue matrix of the COAD
        myerson_revenue_matrix: revenue matrix of the Empirical Myerson Auction
        second_price: average revenue of the second-price auction
        max_welfares: average values of the maximum social welfare
        coverage_matrix: coverage probability matrix of the conformal prediction interval
        Reserve: the average single reserve price of the Empirical Myerson mechenism in each auction
    '''
    
    l = len(Table_new)

    COAD_matrix = np.zeros((num_runs, l))
    myerson_revenue_matrix = np.zeros((num_runs, l))
    second_price_matrix = np.zeros((num_runs, l))
    max_welfares_matrix = np.zeros((num_runs, l))
    coverage_matrix = np.zeros((num_runs, l))
    Reserve_matrix = np.zeros((num_runs, l))

    for i in range(num_runs):
        Pay, new_data_myerson, new_data_2nd, welfare, IR, Reserve_Price = single_exp(Table_old, Table_new, N, item)
        COAD_matrix[i, :] = Pay
        myerson_revenue_matrix[i, :] = new_data_myerson
        second_price_matrix[i, :] = new_data_2nd
        max_welfares_matrix[i, :] = welfare
        coverage_matrix[i, :] = IR
        Reserve_matrix[i, :] = Reserve_Price

    second_price = np.mean(second_price_matrix, axis=0)
    max_welfares = np.mean(max_welfares_matrix, axis=0)
    Reserve = np.mean(Reserve_matrix, axis=0)
    return COAD_matrix, myerson_revenue_matrix, second_price, max_welfares, coverage_matrix, Reserve  

######################################################################################################################


def get_CI(data):
    '''
    Calculates the 95% confidence interval bounds for the mean of the input data to add the error bars
    '''
    lower_percentile = np.mean(data)- 1.96*np.std(data)/np.sqrt(len(data))
    upper_percentile = np.mean(data)+ 1.96*np.std(data)/np.sqrt(len(data))
    return lower_percentile, upper_percentile


def diff_N(Table_his, Table_new, N=500, item="syschannel", num_runs=10, indices=[0]):
    '''
    For a fixed type of item, calculate the average revenue of different mechanisms 
    and the coverage probabilities of the conformal prediction intervals 
    in the corresponding auction using N data points based on "num_runs" experiments
    '''
    COAD_matrix, myerson_revenue_matrix, second_price, max_welfares, coverage_matrix, Reserve = repeated_exp(Table_his, Table_new,N,item,num_runs)
    COAD_matrix = COAD_matrix[:, indices]
    myerson_revenue_matrix = myerson_revenue_matrix[:, indices]
    second_price = [second_price[i] for i in indices]
    max_welfares = [max_welfares[i] for i in indices]
    coverage_matrix = coverage_matrix[:, indices]
    Reserve = [Reserve[i] for i in indices]
    
    COAD = np.mean(COAD_matrix, axis=0)
    myerson_revenue = np.mean(myerson_revenue_matrix, axis=0)

    COAD_average = np.mean(COAD)
    COAD_CI = get_CI(np.mean(COAD_matrix, axis=1))
    myerson_revenue_average = np.mean(myerson_revenue)
    myerson_revenue_CI = get_CI(np.mean(myerson_revenue_matrix, axis=1))
    second_price_average = np.mean(second_price)
    max_welfares_average = np.mean(max_welfares)
    reserve_price = np.mean(Reserve)

    coverage_IR = np.mean(coverage_matrix, axis=1)
    return COAD_average, COAD_CI, myerson_revenue_average, myerson_revenue_CI, second_price_average, max_welfares_average,coverage_IR, reserve_price


