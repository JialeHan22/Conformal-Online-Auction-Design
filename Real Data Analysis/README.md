# Real Data Analysis on eBay Data in Section 6

This folder contains the implementation and experiment code associated with our study on comparing the revenue performance of various auction mechanisms and evaluating the coverage probability of the conformal prediction interval using real-world data from eBay.

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
  Implements the empirical Myerson auction following the methodology of Cole and Roughgarden (2014).

* **`conformal_auxiliary.py`**
  Contains auxiliary functions for the conformal prediction framework, used primarily to identify the item type in each auction.

* **`real_data_exp.py`**
  Implements the core experimental setup. This module conducts empirical comparisons of the expected revenue under different auction mechanisms, and also calculates the coverage probability of the conformal prediction interval for the true value.

* **`split_table.py`**
  Utility module for randomly splitting historical auction data into training and calibration sets. 

### Dataset

* **`Palm+7-day+149auctions+Curve+Clustering.csv`**
  The original auction dataset collected from https://www.modelingonlineauctions.com/datasets. 

## Usage

To reproduce our results:

1. Ensure that all `.py` files are in the working directory or your Python path.
2. Launch and run the Jupyter notebook `new_eBay_real_data_code.ipynb`.
3. All intermediate results (e.g., figures, boxplots) will be generated automatically.


