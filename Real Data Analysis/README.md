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
  Contains auxiliary functions for the conformal prediction framework, used primarily to identify the item type in each auction.

* **`real_data_exp.py`**
  This module implements the core experimental setup, providing functions to conduct empirical comparisons of expected revenue across different auction mechanisms and to compute the coverage probability of conformal prediction intervals for the true valuation.

* **`split_table.py`**
  This module provides functions for randomly partitioning historical auction data into training and calibration sets.

### Dataset

* **`Palm+7-day+149auctions+Curve+Clustering.csv`**
  The original auction dataset collected from https://www.modelingonlineauctions.com/datasets. 

## Usage

To reproduce our results:

1. Ensure that all `.py` files are in the working directory or your Python path.
2. Launch and run the Jupyter notebook `eBay_real_data_code.ipynb`.
3. All intermediate results (e.g., figures, boxplots) will be generated automatically.


