# Real Data Analysis on eBay Data in Section 6

This folder contains the implementation and experimental code for the real-data analysis presented in Section 6, which compares the revenue performance of various auction mechanisms and evaluates the coverage probability of conformal prediction intervals using real-world data from eBay.

## Contents

### Main Files

* **`eBay_real_data_code.ipynb`**
  This is the main Jupyter notebook used to run the full pipeline. It includes:

  * Preprocessing of the auction dataset,
  * Revenue evaluation of different auction mechanisms,
  * Coverage probability evaluation of the conformal prediction interval,
  * Generation of all plots used in Section 6.

### Core Modules

* **`Myerson_auction.py`**
  This module provides functions to implement the empirical Myerson auction, following the methodology of Cole and Roughgarden (2014), given historical value data.

* **`conformal_auxiliary.py`**
  This module contains auxiliary functions for the conformal prediction framework, used primarily to identify the item type in each auction.

* **`real_data_exp.py`**
  This module implements the core experimental setup, providing functions to conduct empirical comparisons of expected revenue across different auction mechanisms and to compute the coverage probability of conformal prediction intervals for the true valuation.

* **`split_table.py`**
  This module provides functions for randomly partitioning historical auction data into training and calibration sets.

### Dataset

* **`Palm+7-day+149auctions+Curve+Clustering.csv`**
  The original auction dataset collected from https://www.modelingonlineauctions.com/datasets. 

Data Dictionary:

Auction ID: Identifier (nominal categorical) uniquely identifying each auction.
BidAmount: Continuous numerical variable representing the bid amount (in U.S. dollars) placed by a bidder.
BidTime: Continuous numerical variable measuring the elapsed time (in days) from the start of the auction when the bid was placed.
Bidder: Identifier (nominal categorical) denoting the eBay username of the bidder.
Bidder Rating: Discrete numerical variable representing the eBay feedback rating of the bidder.
Closing Price: Continuous numerical variable indicating the final winning bid (in U.S. dollars) when the auction closes.
Opening Bid: Continuous numerical variable representing the opening bid (in U.S. dollars) set by the seller.
Seller: Identifier (nominal categorical) denoting the eBay username of the seller.
Seller Rating: Discrete numerical variable representing the eBay feedback rating of the seller.
# Bids: Discrete numerical variable indicating the total number of bids submitted in the corresponding auction.
End Date: Categorical variable representing the auction end date, formatted as mm/dd/yyyy.


## Usage

To reproduce our results:

1. Ensure that all `.py` files are in the working directory or your Python path.
2. Launch and run the Jupyter notebook `eBay_real_data_code.ipynb`.
3. All intermediate results (e.g., figures, boxplots) will be generated automatically.


