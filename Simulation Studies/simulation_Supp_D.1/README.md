# Additional Numerical Results in Supplementary Materials Section D.1

This folder contains the implementation and experimental code for our study in Supplementary Materials Section D.1, which compares the revenue performance of various auction mechanisms and evaluates the coverage probability of conformal prediction intervals through a simulation study where several key assumptions of our framework are not satisfied.

## Contents

### Main Files

* **`simulation_Supp_D.1.ipynb`**
  This is the main Jupyter notebook used to run the full pipeline. It includes:

  * Coverage probability evaluation of the conformal prediction interval,
  * Revenue evaluation of different auction mechanisms,
  * Generation of all plots used in Section D.1.

### Core Modules

* **`Myerson_auction.py`**
  This module provides functions to implement the empirical Myerson auction, following the methodology of Cole and Roughgarden (2014), given historical value data.

* **`conformal_auxiliary.py`**
  Contains auxiliary functions for the conformal prediction framework, used primarily to identify the item type in each auction.
  
* **`coverage.py`**
  This module provides functions to calculate the coverage probability of the conformal prediction interval for the true value.

* **`simu_exp.py`**
  Implements the core experimental setup. This module conducts empirical comparisons of the expected revenue under different auction mechanisms.


## Usage

To reproduce our results:

1. Ensure that all `.py` files are in the working directory or your Python path.
2. Launch and run the Jupyter notebook `simulation_Supp_D.1.ipynb`.
3. All intermediate results (e.g., figures, boxplots) will be generated automatically.


