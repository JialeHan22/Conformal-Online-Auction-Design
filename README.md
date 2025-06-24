# Conformal Online Auction Design

We provide the implementation codes in Python and present the corresponding results in Jupyter Notebook files.

Each experiment is contained in a separate folder. These folders include all relevant Python modules (`.py` files) and corresponding Jupyter Notebook files (`.ipynb`) for executing and reproducing the results. In addition, each experiment folder contains a dedicated `README.md` file that documents the purpose and usage of the associated modules and notebooks.

## Folder Structure

* The **"Real Data Analysis"** folder contains the real-world eBay dataset obtained from [https://www.modelingonlineauctions.com/datasets](https://www.modelingonlineauctions.com/datasets), and reproduces the analysis from Section 6 of our paper.
* The **"Simulation Studies"** folder includes four independent subfolders:

  * `simulation_Section_7.1`
  * `simulation_Section_7.2`
  * `simulation_Supp_D.1`
  * `simulation_Supp_D.2`

Each subfolder corresponds to a distinct experiment from Section 7 of the main paper or Section D of the Supplementary Material.

## Reproducibility

The experiments are designed to be reproducible across platforms.

### High-Performance Computing Environment

Experiments were primarily executed on a high-performance computing cluster under the following specifications:

* **CPU architecture**: Intel Gold
* **Python version**: 3.9.6
* **Memory per core**: 32 GB
* **Execution interface**: Jupyter Notebook

All required Python packages are listed in `requirements.txt`, generated via `pip freeze`. This ensures full reproducibility of our results on compatible systems.

### Local Execution

All experiments can also be successfully executed on a personal computer. The notebooks have been tested using Python 3.10.13 on a Mac equipped with Apple Silicon (M4 Pro). A list of essential packages for local execution is provided in `Mac_requirements.txt`.




