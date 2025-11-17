# Near-optimal Linear Predictive Clustering in Non-separable Spaces via Mixed Integer Programming and Quadratic Pseudo-Boolean Reductions

The code and experiment results for the paper "Near-optimal Linear Predictive Clustering in Non-separable Spaces via Mixed Integer Programming and Quadratic Pseudo-Boolean Reductions", accepted in AAAI-26.

For the full paper with appendix, please refer to `extended_paper.pdf`
## Abstract
Linear Predictive Clustering (LPC) partitions samples based on shared linear relationships between feature and target variables, with numerous applications including marketing, medicine, and education. Greedy optimization methods, commonly used for LPC, alternate between clustering and linear regression but lack global optimality. While effective for separable clusters, they struggle in *non-separable* settings where clusters overlap in feature space.  In an alternative constrained optimization paradigm, Bertsimas & Shioda (2007) formulated LPC as a Mixed-Integer Program (MIP), ensuring global optimality regardless of separability but suffering from poor scalability.  This work builds on the constrained optimization paradigm to introduce two novel approaches that improve the efficiency of global optimization for LPC. By leveraging key theoretical properties of separability, we derive near-optimal approximations with provable error bounds, significantly reducing the MIP formulation’s complexity and improving scalability. Additionally, we can further approximate LPC as a Quadratic Pseudo-Boolean Optimization (QPBO) problem, achieving additional computational gains in the special case of two clusters. Comparative analyses on synthetic and real-world datasets demonstrate that our methods consistently achieve near-optimal solutions with substantially lower regression errors than greedy optimization while exhibiting superior scalability over existing MIP formulations.

## Code Structure  

The repository is organized as follows:  

- `scripts/` contains the implementation of the proposed methods and baselines.  
    - `models.py` implements LPC-NS-MIP, LPC-NS-QPBO, and GlobalOpt.  
    - `Greedy_Codes/` includes the implementation of Greedy optimization methods.  
- `synthetic_data/` provides code for generating synthetic datasets.  
- `real_data/` contains preprocessing scripts for real-world datasets.  
    - `clean_data_collection/` includes the cleaned datasets used in experiments. Oranized by (UCI-ID_CategoricalFeatureNameForPartitioning).
- `evaluation/` contains performance evaluation scripts.  
    - `eval.py` evaluates regression error, clustering error, and runtime.  
    - `utils.py` provides utility functions for error calculation and result loading.  
    - `visual.py` generates visualizations of experimental results.  
- `experiments/` includes scripts to run experiments.  
    - `experiment_scripts/` contains scripts for experiment reproducibility.  
    - `results/` stores precomputed experimental results.  
- `experiment_results_visualization.ipynb` visualizes results using precomputed data.  
- `solver_performance_comparison.ipynb` compares performance across SCIP, Gurobi-OPB, and Gurobi-MIP solvers.  

## Experiment Results and Reproducibility  

Note: The experiments take a long time (2 hrs per seed and setting) to run; the pre-computed results are available in the `experiments/results` folder.
To visualize the results, run the [experiment_results_visualization.ipynb](experiment_results_visualization.ipynb)

To reproduce the experimental results:  

1. Clone the repository.  
2. Install dependencies: `pip install -r requirements.txt`.  
3. Obtain a valid Gurobi license (required for MIP optimization).  
4. Run the experiment scripts in `experiments/experiment_scripts/`, organized by research questions:
    - **RQ1: Time vs. Error Scalability Analysis** [rq1_time_vs_error_scalability_analysis.py](experiments/experiment_scripts/rq1_time_vs_error_scalability.py)
    - **RQ2: Performance Analysis on Synthetic Datasets** 
        - with Varying Noise Levels: [rq2_performance_analysis_on_synthetic_datasets-noise_std.py](experiments/experiment_scripts/rq2_performance_analysis_on_synthetic_datasets-noise_std.py)
        - with Varying Dimension of Features: [rq2_performance_analysis_on_synthetic_datasets-dimension.py](experiments/experiment_scripts/rq2_performance_analysis_on_synthetic_datasets-dimension.py)
    - **RQ3: Performance Analysis on Real-World Datasets** [rq3_real_dataset.py](experiments/experiment_scripts/rq3_real_dataset.py)

5. Additional experiments are included in the papers:
    - $\mathcal{E}_{sep}$ analysis and approximation error of $\mathbf{w}^\ast$ in Table 1: [approximation_error_analysis.py](experiments/experiment_scripts/approximation_error_analysis.py)
    - Appendix E: Performance Comparison between solvers [appendix_pbo_solver_performance_comparsion.py](experiments/experiment_scripts/appendix_pbo_solver_performance_comparsion.py) with results in [solver_performance_comparison.ipynb](solver_performance_comparsion.ipynb)
    - Appendix F: Additional Sythetics Experiment
       - with Varying Proportion of Outliers: [rq2_performance_analysis_on_synthetic_datasets-outlier.py](experiments/experiment_scripts/rq2_performance_analysis_on_synthetic_datasets-outlier.py)
