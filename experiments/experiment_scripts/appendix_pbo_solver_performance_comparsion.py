import sys
import os
import json
import time
import numpy as np
import json
import time 
import matplotlib as mpl
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import files from repository
from synthetic_data.data import SyntheticData
from  synthetic_data.cluster_size import *
from  synthetic_data.outlier import *
from  synthetic_data.cluster_label import *
from  synthetic_data.regression_params import *
from  synthetic_data.target import *
from scripts.model import *


def compare_miqp_pbo(n_values=[20,70,100,200],
                     p=1, K=2, lambda_reg=0.7, noise_std=1.5,
                     time_limit=10800, save_filename="results.json"):
    """
    Compares runtime performance between MIQP and PBO solvers.
    
    Args:
        n_values (list): Sample sizes to test [default: [20,70,100,200]]
        p (int): Number of features [default: 1]
        K (int): Number of clusters [default: 2]
        lambda_reg (float): Regularization parameter [default: 0.7]
        noise_std (float): Standard deviation of noise [default: 1.5]
        time_limit (int): Maximum runtime in seconds [default: 10800]
        save_filename (str): Filename to save results [default: "results.json"]
        
    Returns:
        dict: Results containing timing data for each solver
    """

    os.makedirs("OPB results", exist_ok=True)
    
    gurobi_mip_times = []    
    gurobi_pbo_times = []
    scip_pbo_times = []

    for idx, n in enumerate(n_values):
        # Generate synthetic data once and share across models
        data = SyntheticData(random_state=42)
        data.generate_data(N=n, K=K, D=p, noise_std=noise_std)

        # --- 1) LPC-NS-QPBO ---
        qpbo_model = LpcNsQbpo(
            n=n, p=p, K=K, lambda_reg=lambda_reg,
            noise_std=noise_std, random_state=42, verbose=False,
            loss='QBPO'
        )
        qpbo_model.X = data.X
        qpbo_model.y = data.y
        qpbo_model.prepare_augmented_data()
        
        start_app = time.time()
        qpbo_model.perform_gurobi_miqp()
        qpbo_model.compute_cluster_weights()
        end_app = time.time()
        gurobi_mip_times.append(end_app - start_app)


        # --- 2) LPC-NS-QPBO-OPB-Gurobi ---
        gurobi_pbo_model = PBOClusterRegressionModel(
            n=n, p=p, K=K, lambda_reg=lambda_reg,
            noise_std=noise_std, random_state=42, verbose=False
        )
        gurobi_pbo_model.X = data.X 
        gurobi_pbo_model.y = data.y
        gurobi_pbo_model.prepare_augmented_data()  # Add this line
        
        
        pbo_filename = f"results/pbo/opb_files/pbo_{n}.opb"
        os.makedirs('results/pbo/opb_files', exist_ok=True)

        gurobi_pbo_model.generate_opb_file(pbo_filename)
        start_gurobi_pbo = time.time()
        gurobi_pbo_model.solve_opb_model(opb_filename=pbo_filename, solver='gurobi')
        gurobi_pbo_model.compute_cluster_weights()
        end_gurobi_pbo = time.time()
        gurobi_pbo_times.append(end_gurobi_pbo - start_gurobi_pbo)


        # --- 3) LPC-NS-QPBO-OPB-SCIP (only for n <= 100) ---
        if n <= 100:
            scip_pbo_model = PBOClusterRegressionModel(
                n=n, p=p, K=K, lambda_reg=lambda_reg,
                noise_std=noise_std, random_state=42, verbose=False
            )
            scip_pbo_model.X = data.X 
            scip_pbo_model.y = data.y

            scip_pbo_model.prepare_augmented_data()

            # pbo_filename = f"pbo_{n}.opb"
            # scip_pbo_model.generate_opb_file(pbo_filename)

            start_scip_pbo = time.time()
            scip_pbo_model.solve_opb_model(opb_filename=pbo_filename, solver='scip')
            scip_pbo_model.compute_cluster_weights()
            end_scip_pbo = time.time()
            scip_pbo_times.append(end_scip_pbo - start_scip_pbo)

        # --- Save results iteratively after each n ---
        current_results = {
            'n_values': n_values,
            'gurobi_mip_times': gurobi_mip_times,
            'gurobi_pbo_times': gurobi_pbo_times,
            'scip_pbo_times': scip_pbo_times
        }
        
        # Write to JSON file results up to the current point
        with open(os.path.join("results/pbo", save_filename), 'w') as f:
            json.dump(current_results, f, indent=4)

    return current_results



def plot_comparison(results):

    """
    Plots runtime comparison between different QPBO implementations.
    
    Args:
        results (dict): Dictionary containing:
            - n_values: List of sample sizes
            - gurobi_mip_times: MIQP solver times 
            - gurobi_pbo_times: Gurobi PBO solver times
            - scip_pbo_times: SCIP solver times
            
    Generates:
        PDF plot saved as 'runtime_comparison_pbo.pdf'
    """ 
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    n_values = results['n_values']
    scip_pbo_times = results['scip_pbo_times']
    gurobi_mip_times = results['gurobi_mip_times']
    gurobi_pbo_times = results['gurobi_pbo_times']

    plt.figure(figsize=(6, 4))

    # Plot LPC-NS-QPBO with unfilled marker
    plt.plot(n_values,
             gurobi_mip_times,
             color='red', marker='x', linestyle='--', linewidth=1.0, markersize=12, markerfacecolor='none', alpha=0.8,
             label='LPC-NS-QPBO-MIP-Gurobi')

    # Plot OPB-Gurobi with unfilled diamond
    plt.plot(n_values[:len(gurobi_pbo_times)],
             gurobi_pbo_times,
             color='#FFA500', marker="D", linestyle='--', linewidth=1.0, markersize=12, markerfacecolor='none', alpha=0.8,
             label='LPC-NS-QPBO-OPB-Gurobi')

    # Plot SCIP with unfilled triangle
    if len(n_values[:len(scip_pbo_times)]) > 0:
        plt.plot(n_values[:len(scip_pbo_times)],
                 scip_pbo_times,
                 color='indigo', marker="v", linestyle='--', linewidth=1.0, markersize=12, markerfacecolor='none', alpha=0.8,
                 label='LPC-NS-QPBO-OPB-SCIP')

    # Time limit line with text label above
    plt.axhline(y=10800, color='red', linestyle='dashed')
    plt.text(n_values[0], 1.15e4, 'Time Limit', color='red')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    all_x = list(n_values)
    all_x = sorted(list(set(all_x)))
    plt.xticks(all_x, labels=[str(int(x)) for x in all_x])

    plt.xlabel('Number of Samples')
    plt.ylabel('Time (s)')
    plt.title('Runtime Comparison of QPBO using a MIP Solver vs. Dedicated PBO Solvers')
    plt.legend()
    plt.tight_layout()

    plt.savefig('results/pbo/runtime_comparison_pbo.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def load_pbo_results(filename):
    """
    Loads PBO benchmark results from JSON file.
    
    Args:
        filename (str): Name of JSON file containing results
        
    Returns:
        dict: Dictionary containing:
            - n_values: List of sample sizes
            - gurobi_mip_times: MIQP solver times
            - gurobi_pbo_times: Gurobi PBO solver times 
            - scip_pbo_times: SCIP solver times
    """
    os.makedirs('results/pbo', exist_ok=True)
    f = os.path.join("results/pbo", filename)

    with open(f, 'r') as f:
        data = json.load(f)
    current_results = {
        'n_values': data['n_values'],
        'gurobi_mip_times': data['gurobi_mip_times'],
        'gurobi_pbo_times': data['gurobi_pbo_times'],
        'scip_pbo_times': data['scip_pbo_times']
    }
    return current_results