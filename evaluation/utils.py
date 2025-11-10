from sklearn.metrics import rand_score, r2_score, mean_squared_error,confusion_matrix, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import contingency_matrix
from sklearn.linear_model import Ridge, QuantileRegressor
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import json
import pandas as pd
from itertools import permutations
#import sem from scipy
from scipy.stats import sem
import scipy.stats as stats
#from Evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
import seaborn as sns

import time
class Utils():
    def __init__(self):
        pass

    @staticmethod
    def mse_r2(y_pred, y_true, error='mse'):
        '''
        Compute the mean squared error and R2 score between the predicted and true labels

        Parameters:
        - y_pred: 1D array-like, predicted labels
        - y_true: 1D array-like, ground truth labels
        '''
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    @staticmethod
    def hungarian_algorithm(true_labels, pred_labels):
        '''
        Compute the label alignment between the cluster assignments and the ground truth assignments

        Parameters:
        - true_labels: 1D array-like, ground truth labels
        - pred_labels: 1D array-like, predicted cluster labels

        Returns:
        - aligned_labels: 1D array-like, predicted labels aligned with ground truth
        - mismatch: int, the number of unequal labels
        - label_mapping: dict, mapping from predicted to true labels
        '''

        # Compute confusion matrix
        true_unique = np.unique(true_labels)
        pred_unique = true_unique

        cost_matrix = confusion_matrix(true_labels, pred_labels)

        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        
        # Create a mapping from predicted to true labels
        label_mapping = {pred_unique[row]: true_unique[col] for row, col in zip(row_ind, col_ind)}
        
        # Relabel the predicted labels
        aligned_labels = np.array([label_mapping[label] for label in pred_labels])

        # Compute the mismatch as the number of unequal labels
        mismatch = np.sum(true_labels != aligned_labels)
        
        return aligned_labels, mismatch, label_mapping

    @staticmethod
    def clustering_label_alignment(true_labels, pred_labels, check_unique=True):
        """
        Align cluster labels with ground truth labels by evaluating all possible alignments.
        
        Parameters:
        - pred_labels: 1D array-like, predicted cluster labels
        - true_labels: 1D array-like, ground truth labels
        
        Returns:
        - best_aligned_labels: 1D array-like, predicted labels aligned with ground truth
        - min_mismatch: int, the smallest mismatch achieved
        """
        # Ensure inputs are numpy arrays
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        
        # Get unique predicted and true labels
        pred_unique = np.unique(pred_labels)
        true_unique = np.unique(true_labels)
        if len(pred_unique) > len(true_unique) and check_unique:
            raise ValueError("Predicted labels have more unique classes than true labels, which is unexpected.")
        
        # Initialize variables to track the best alignment
        min_mismatch = float('inf')
        best_mapping = None
        
        # Generate all permutations of the ground truth labels matching the number of predicted labels
        for perm in permutations(true_unique, len(pred_unique)):
            # Create a mapping from predicted labels to this permutation of true labels
            mapping = {pred: true for pred, true in zip(pred_unique, perm)}
            
            # Apply the mapping to predicted labels
            aligned_labels = np.array([mapping[label] for label in pred_labels])
            
            # Compute the mismatch as the number of unequal labels
            mismatch = np.sum(aligned_labels != true_labels)
            
            # Update the best mapping if this alignment has fewer mismatches
            if mismatch < min_mismatch:
                min_mismatch = mismatch
                best_mapping = mapping
        
        # Apply the best mapping to predicted labels
        best_aligned_labels = np.array([best_mapping[label] for label in pred_labels])
        
        return best_aligned_labels, min_mismatch, best_mapping


    @staticmethod
    def label_mismatch(pred_labels, true_labels):
        '''
        Compute the label mismatch between the cluster assignments and the ground truth assignments
        '''

        # align labels
        aligned_pred_labels, min_mismatch , _ = Utils.clustering_label_alignment(true_labels, pred_labels)
        #calculate mismatch
        mismatch_aligned = np.sum(true_labels != aligned_pred_labels)
        mismatch = np.sum(true_labels != pred_labels)
        return np.min([mismatch_aligned, mismatch])/len(true_labels)
    @staticmethod
    def weight_mismatch(regression_weights, ground_truth_weights, pred_labels, true_labels, check_unique=True):
        '''
        Compute the weight mismatch between the regression weights and the ground truth weights
        '''
        # compute the norm in ground truth weights
        aligned_labels, _, label_mapping = Utils.clustering_label_alignment(true_labels, pred_labels, check_unique=check_unique)
        
        difference = 0
        for k, pred_class in enumerate(np.unique(pred_labels)):
            #obtain the index of first occurence of label in pred_labels
            ground_truth_class = label_mapping[pred_class]
            difference += np.linalg.norm(regression_weights[pred_class] - ground_truth_weights[ground_truth_class])
        return difference
    
    @staticmethod
    def load_evaluation_result(file_path, optimal_folder_path = None):
        '''
        Load evaluation results from given file path
        '''
        path_dict = {}
        optimal_parameter_lst = os.listdir(optimal_folder_path)
        for parameter in optimal_parameter_lst:
            if parameter == ".DS_Store":
                continue
            dataset_num_list = os.listdir(os.path.join(optimal_folder_path, parameter))
            for dataset_num in dataset_num_list:
                random_states_list = os.listdir(os.path.join(optimal_folder_path, parameter, dataset_num))
                for random_states in random_states_list:
                    if optimal_folder_path is not None:
                        opath = os.path.join(optimal_folder_path, parameter, dataset_num, random_states)
                        if os.path.exists(opath):
                            path_dict[parameter] = opath

        parameter_lst = os.listdir(file_path)
        results_df = pd.DataFrame()
        for parameter in parameter_lst :
            if parameter == ".DS_Store":
                continue
            parameter_val = float(parameter.split("=")[1])
            dataset_num_list = os.listdir(os.path.join(file_path, parameter))
            random_state_df = pd.DataFrame()
            for random_state in dataset_num_list:
                if random_state == ".DS_Store":
                    continue
                dataset_num = float(random_state.split("=")[1])
                lambda_list = os.listdir(os.path.join(file_path, parameter, random_state))
                lambda_list = np.sort(lambda_list)
                lambda_dfs = pd.DataFrame()
                for current_lambda in lambda_list:
                    if current_lambda == ".DS_Store":
                        continue
                    path = os.path.join(file_path, parameter, random_state, current_lambda, "evaluation_results.json")
                    with open(os.path.join(file_path, parameter, random_state, current_lambda, "regression_weights.json")) as f:
                        regression_weights = json.load(f)
                    with open(os.path.join(file_path, parameter, random_state, current_lambda, "cluster_assignments.json")) as f:
                        cluster_assignments = json.load(f)
                    with open(os.path.join(file_path, parameter, random_state, current_lambda, "data.json")) as f:
                        data = json.load(f)
                        X = np.array(data['X'], dtype=float)
                        y = np.array(data['y'], dtype=float)
                        z = np.array(data['z'], dtype=int)
                        train_indices = np.array(data['train'], dtype=int)
                        X = X[train_indices]
                        y = y[train_indices]
                        z = z[train_indices]

                    if optimal_folder_path is not None:
                        opath = os.path.join(optimal_folder_path, parameter, random_state, current_lambda)
                        if not os.path.exists(opath) and parameter in path_dict.keys():
                            opath = path_dict[parameter]
                        elif not os.path.exists(opath):
                            opath = None
                        if opath is not None:
                            with open(os.path.join(opath, "regression_weights.json")) as f:
                                optimal_regression_weights = np.array(json.load(f)['milp']).astype(float)
                            with open(os.path.join(opath, "cluster_assignments.json")) as f:
                                optimal_cluster_assignments = np.array(json.load(f)['milp']).astype(int)
                            with open(os.path.join(opath, "evaluation_results.json")) as f:
                                optimal_eveluation_result = json.load(f)       

                    eveluation_result = json.load(open(path))
                    lambda_val = float(current_lambda.split("=")[1])
                    lambda_df = pd.DataFrame(eveluation_result, index=[lambda_val])
                    regression_weight_ground_truth = np.array(regression_weights['ground_truth'], dtype=float)
                    cluster_assignment_milp = np.array(cluster_assignments['milp'], dtype=int)
                    if 'greedy' in cluster_assignments.keys():
                        cluster_assignment_greedy = np.array(cluster_assignments['greedy'], dtype=int)
                    else:
                        cluster_assignment_greedy = None

                    if optimal_folder_path is not None and opath is not None:
                        lambda_df = Utils.evaluate_clustering(X, y, z, cluster_assignment_milp, file_path, 
                                                              {'random_state': lambda_val, 'datanum': int(random_state.split('=')[1]), 'parameter_value': lambda_val},
                                                eveluation_result, regression_weight_ground_truth, optimal_folder_path, 
                                                optimal_regression_weights, optimal_cluster_assignments, optimal_eveluation_result,
                                                cluster_assignment_greedy=cluster_assignment_greedy)
                    else:
                        lambda_df = Utils.evaluate_clustering(X, y, z, cluster_assignment_milp, file_path, 
                                                              {'random_state': lambda_val, 'datanum': int(random_state.split('=')[1]), 'parameter_value': lambda_val}, 
                                                eveluation_result, regression_weight_ground_truth,
                                                cluster_assignment_greedy=cluster_assignment_greedy)
                    lambda_dfs = pd.concat([lambda_dfs, lambda_df])
                lambda_dfs.reset_index(inplace = True)
                lambda_dfs.rename(columns = {'index': 'lambda'}, inplace = True)
                best_lambda = lambda_dfs.mean().to_frame().T
                for col in best_lambda.columns:
                    if col != 'lambda':
                        best_lambda[col] = str(best_lambda[col].values[0]) + "±" + str(Utils.standard_error(lambda_dfs[col]))
                best_lambda['dataset'] = dataset_num
                random_state_df = pd.concat([random_state_df, best_lambda])
            random_state_df.reset_index(drop = True, inplace = True)
            #average_random_state = random_state_df.mean().to_frame().T
            random_state_df['parameter'] = parameter_val
            results_df = pd.concat([results_df, random_state_df])
        parameter_name = [i for i in parameter_lst if i != ".DS_Store"][0].split("=")[0]
        results_df.reset_index(inplace = True, drop=True)
        results_df.rename(columns = {'parameter': parameter_name}, inplace = True)
        results_df.sort_values(by = parameter_name, inplace = True)
        #move the parameter column to the front
        cols = results_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        results_df = results_df[cols]
        ressults_df_lst = []
        meta_data = file_path.split("/")[-1].split("_")
        for meta in meta_data:
            if "N=" in meta:
                N = int(meta.split("=")[1])
            if 'D=' in meta:
                D = int(meta.split("=")[1])
            if 'K=' in meta:
                K = int(meta.split("=")[1])
                
        for random_state in results_df['dataset'].unique():
            random_state_df = results_df[results_df['dataset'] == random_state]
            #random_state_df['meta'] = "Number of data points: " + str(N) + ", Dimension: " + str(D) + ", Number of clusters: " + str(K)
            ressults_df_lst.append(random_state_df.reset_index(drop = True))

        return ressults_df_lst
    

    
    @staticmethod
    def load_computer_canada_result(parameter, folder_name, list_of_parameters, optimal_folder_path = None):
        '''
        An helper function to read the results of the experiments conducted on Compute Canada 
        
        parameter: str: the parameter that was varied in the experiments
        folder_name: str: the name of the folder containing the results
        list_of_parameters: list: a list of the parameter values that were varied in the experiments to prevent missing values caused by the computer canada system fa
        optimal_folder_path: str: the name of the folder containing the optimal results to calculate the difference between the optimal results and the results from the experiments

        return:
        a dataframe containing the best results for each parameter value based on the evaluation metrics MSE
        '''
        
        results_df = pd.DataFrame()
        input_path = f'{folder_name}/input/input_parameters_Predicted_Clustering'
        output_path = f'{folder_name}/output/Predicted_Clustering'
        parameter_path = f'{folder_name}/input_parameters_Predicted_Clustering.txt'
        if optimal_folder_path is not None:
            optimal_folder_path = f'{optimal_folder_path}/output/Predicted_Clustering'
        with open(parameter_path, 'r') as f:
            lines = f.readlines()
            list_of_parameters = lines[2]
            list_of_parameters = list_of_parameters.split(":")[1].strip().split(",")
            list_of_parameters = [float(i.strip()) for i in list_of_parameters]
        input_lst = os.listdir(input_path)
        for i, input_file in enumerate(input_lst):
            if input_file.endswith('.txt'):
                input_id = input_file.split('-')[-1].split('.')[0]
                lines = open(f'{input_path}/{input_file}', 'r').readlines()[1:]
                input_params = {}
                for line in lines:
                    key = line.split(':')[0]
                    value = line.split(':')[1].strip()
                    input_params[key] = value
                result = f'{output_path}/{input_id}/evaluation_results_{input_id}.json'
                cluster_assignment = f'{output_path}/{input_id}/cluster_assignments_{input_id}.json'
                regression_weights = f'{output_path}/{input_id}/regression_weights_{input_id}.json'
                data = f'{output_path}/{input_id}/data_{input_id}.json'
                if optimal_folder_path is not None and os.path.exists(f'{optimal_folder_path}/{input_id}'):
                    optimal_regression_weights = f'{optimal_folder_path}/{input_id}/regression_weights_{input_id}.json'
                    optimal_cluster_assignments = f'{optimal_folder_path}/{input_id}/cluster_assignments_{input_id}.json'
                    optimal_evaluation_results = f'{optimal_folder_path}/{input_id}/evaluation_results_{input_id}.json'

                if os.path.exists(result):
                    with open(result, 'r') as f:
                        evaluation_results = json.load(f)
                    with open(cluster_assignment, 'r') as f:
                        cluster_assignment = json.load(f)
                        cluster_assignment_milp = np.array(cluster_assignment['milp'], dtype=int)
                    with open(data, 'r') as f:
                        data = json.load(f)
                        y = np.array(data['y'], dtype=float)
                        X = np.array(data['X'], dtype=float) 
                        z = np.array(data['z'], dtype=int)
                        K = np.max(z) + 1
                        if 'train' in data.keys():
                            train_indices = np.array(data['train'], dtype=int)
                            X = X[train_indices]
                            X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))
                            U, S, Vt = np.linalg.svd(X_augmented, full_matrices=False)
                            XV = X_augmented @ Vt.T @ np.linalg.inv(np.diag(S))
                            y = y[train_indices]
                            z = z[train_indices]
                    with open(regression_weights, 'r') as f:
                        weight_lst = json.load(f)
                        regression_weight_ground_truth = np.array(weight_lst['ground_truth'], dtype=float)
                    if optimal_folder_path is not None and os.path.exists(f'{optimal_folder_path}/{input_id}'):
                        with open(optimal_regression_weights, 'r') as f:
                            regression_weight_optimal = np.array(json.load(f)['milp'], dtype=float)
                        with open(optimal_cluster_assignments, 'r') as f:
                            optimal_cluster_assignment = np.array(json.load(f)['milp'], dtype=int)
                        with open(optimal_evaluation_results, 'r') as f:
                            optimal_evaluation_results = json.load(f)
                    if 'greedy' in cluster_assignment.keys():
                        cluster_assignment_greedy = np.array(cluster_assignment['greedy'], dtype=int)
                    else:
                        cluster_assignment_greedy = None
                    
                    if optimal_folder_path is not None and os.path.exists(f'{optimal_folder_path}/{input_id}'):
                        result_df = Utils.evaluate_clustering(X, y, z, cluster_assignment_milp, folder_name, input_params, 
                                                evaluation_results, regression_weight_ground_truth, optimal_folder_path, 
                                                regression_weight_optimal, optimal_cluster_assignment, optimal_evaluation_results,
                                                cluster_assignment_greedy=cluster_assignment_greedy)
                    else:
                        result_df = Utils.evaluate_clustering(X, y, z, cluster_assignment_milp, folder_name, input_params, 
                                                evaluation_results, regression_weight_ground_truth,
                                                cluster_assignment_greedy=cluster_assignment_greedy)
                    if result_df['time_refit_milp_assignment'].values[0] > 36000: # Time limit exceeded
                        result_df['time_refit_milp_assignment'] = 36000
                        result_df['time_milp'] = 36000
                    results_df = pd.concat([results_df, result_df])

        results_df.reset_index(inplace=True, names=parameter)
        mean_random = results_df.groupby([parameter, 'dataset']).mean().reset_index()
        all_datasets = []
        for dataset in np.unique(mean_random['dataset']):
            per_dataset = mean_random[mean_random['dataset'] == dataset]
            flag = False
            for p in list_of_parameters:
                if p not in per_dataset[parameter].values:
                    if parameter == 'N' and not flag:
                        tem = {column: i for i, column in zip(per_dataset.iloc[-1,:].values, per_dataset.columns) if column not in [parameter, 'dataset', 'time']}
                        tem['time_milp'] = 3600
                        flag = True
                    else:
                        tem = {i: np.inf for i in per_dataset.columns if i not in [parameter, 'dataset']}
                    tem[parameter] = p
                    tem['dataset'] = dataset
                    per_dataset = pd.concat([per_dataset, pd.DataFrame(tem, index=[0])])
                for col in per_dataset.columns:
                    if col not in [parameter, 'dataset']:
                        tem = results_df[(results_df[parameter] == p) & (results_df['dataset'] == dataset)][col]
                        per_dataset.loc[per_dataset[parameter] == p, col] = str(per_dataset.loc[per_dataset[parameter] == p, col].values[0]) + '±' + str(round(Utils.standard_error(tem), 2))
            per_dataset = per_dataset.sort_values(parameter).reset_index(drop=True)
            per_dataset = per_dataset[per_dataset[parameter].isin(list_of_parameters)]
            all_datasets.append(per_dataset)
        return all_datasets

    @staticmethod
    def plot_regression_results(data, regression_weights = None, cluster_assignment = None, fig = None, axes = None, **kwargs):
        '''
        Plot fitted regression lines on the data points with the given regression weights and cluster assignments

        Parameters:
        data: SyntheticData | (future: RealData)
            The data object

        regression_weights: list | None
            The list of regression weights for each cluster to be plotted

        cluster_assignment: np.array | None
            The cluster assignment for each data point, else the ground truth cluster assignment in data object will be used

        fig: plt.Figure | None 
            The figure to plot on if provided, else a new figure will be created
        
        axes: plt.Axes
            The axes to plot on if provided, else a new axes will be created
        '''

        if fig is None and axes is None:
            fig, axes = Evaluation.plot_data_with_cluster_label(data, label = cluster_assignment)

        if regression_weights is not None:
            for d in range(data.D):
                x_min = min(data.X[:, d])
                x_max = max(data.X[:, d])
                x_space = np.linspace(x_min, x_max, 100)
                for k in range(len(regression_weights)):
                    weight_by_cluster = regression_weights[k]
                    y_hat = weight_by_cluster[0] + weight_by_cluster[d+1]*x_space
                    if data.D == 1:
                        plt.plot(x_space, y_hat, c =sns.color_palette()[k], **kwargs)
                    else:
                        axes[d].plot(x_space, y_hat, c =sns.color_palette()[k], **kwargs)
        return fig, axes

    @staticmethod
    def prepare_augmented_data(X, whiten=True):
        """
        Adds a column of ones to X for the bias term and computes SVD and related matrices.
        """
        if X is None:
            raise ValueError("Data not generated yet. Call generate_synthetic_data() first.")
        return X
    
    @staticmethod
    def perform_refit_sklearn_ridge_regression(X,y, K, z, loss = 'MSE', lambda_reg = 1):
        """
        Performs Ridge Regression using scikit-learn to refit the regression model within each cluster using the cluster assignments from Gurobi MIQP.

        Parameters:
        if ground_truth is True, the cluster assignments will be based on the true cluster assignments
        
        """

        # refit the regression model using the cluster assignments from Gurobi MIQP
        cluster_assignments = z
        
        w_star_refit = [np.zeros(X.shape[1] + 1) for _ in range(K)]

        for k in range(K):
            # retrieve the data points for the cluster k
            indices = np.where(cluster_assignments == k)[0]
            if len(indices) > 0: # prevent empty clusters
                X_k = X[indices]
                y_k = y[indices]
                # Use Ridge Regression for MSE
                ridge_model_k = Ridge(alpha=lambda_reg, fit_intercept=True)
                ridge_model_k.fit(X_k, y_k)  # Using original X (without the column of ones)
                w_star_refit[k] = np.hstack((ridge_model_k.intercept_, ridge_model_k.coef_)) # [bias, weights]
        return w_star_refit

    @staticmethod
    def top_eigenvector_distance(A, B):
        # Compute eigenvectors and eigenvalues
        _, v_A = np.linalg.eig(A @ A.T)
        _, v_B = np.linalg.eig(B @ B.T)

        # Select the top eigenvector (first column for largest eigenvalue)
        top_v_A = v_A[:, 0]
        top_v_B = v_B[:, 0]

        # Normalize the eigenvectors
        top_v_A /= np.linalg.norm(top_v_A)
        top_v_B /= np.linalg.norm(top_v_B)

        # Compute cosine similarity
        cosine_similarity = np.abs(np.dot(top_v_A, top_v_B))

        # Compute Euclidean distance
        euclidean_distance = np.linalg.norm(top_v_A - top_v_B)

        return cosine_similarity
    

    @staticmethod
    def evaluate_clustering(
        X, y, z, cluster_assignment_milp, folder_name, 
        input_params, evaluation_results, regression_weight_ground_truth, 
        optimal_folder_path=None, regression_weight_optimal=None, 
        optimal_cluster_assignment=None, optimal_evaluation_results=None,
        cluster_assignment_greedy=None
    ):
        """
        Function to evaluate clustering and regression results and return a result dataframe.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - z: Ground truth cluster assignment (numpy array).
        - cluster_assignment_milp: Cluster assignment from MILP (numpy array).
        - folder_name: String indicating the folder context ('MSE' or other).
        - input_params: Dictionary of input parameters.
        - evaluation_results: Dictionary of initial evaluation results.
        - regression_weight_ground_truth: Ground truth regression weights.
        - Utils: Utility functions for weight and label mismatch evaluation.
        - optimal_folder_path: Path to optimal results (optional).
        - regression_weight_optimal: Optimal regression weights (optional).
        - optimal_cluster_assignment: Optimal cluster assignments (optional).

        Returns:
        - result_df: DataFrame containing the evaluation results.
        """
        K = np.max(z) + 1
        V = [np.zeros(X.shape[1] + 1) for _ in range(K)]
        y_pred = np.zeros(X.shape[0])
        regression_weight_milp = [0 for _ in range(K)]
        y_pred_refit_ground_truth = np.zeros(X.shape[0])

        for k in range(K):
            indices = np.where(z == k)[0]
            z_k = np.diag(z == k).astype(int)
            _, _, tem = np.linalg.svd(X[indices], full_matrices=False)
            V[k] = tem.T[0]
            
            refit_time = time.time()
            y_pred_refit_ground_truth[indices] = Ridge(alpha=0.01).fit(X[indices], y[indices]).predict(X[indices])
            refit_time = time.time() - refit_time

            indices = np.where(cluster_assignment_milp == k)[0]
            if indices.shape[0] == 0:
                regression_weight_milp[k] = np.zeros(X.shape[1] + 1)
            else:
                model = Ridge(alpha=0.01)
                y_pred[indices] = model.fit(X[indices], y[indices]).predict(X[indices])
                regression_weight_milp[k] = np.hstack((model.intercept_, model.coef_))

        V_difference = np.linalg.norm(V[1] - V[0])
        result_df = pd.DataFrame(evaluation_results, index=[float(input_params['parameter_value'])])
        result_df['random_state'] = float(input_params['random_state'])
        result_df['dataset'] = int(1)
        result_df['V_difference'] = V_difference
        result_df['refit-weight_mismatch_milp'] = Utils.weight_mismatch(regression_weight_milp, regression_weight_ground_truth, cluster_assignment_milp, z)
        result_df['refit-weight_mismatch_refit_milp_assignment'] = Utils.weight_mismatch(regression_weight_milp, regression_weight_ground_truth, cluster_assignment_milp, z)
        result_df['label_mismatch_milp'] = Utils.label_mismatch(cluster_assignment_milp, z)
        result_df['label_mismatch_refit_milp_assignment'] = Utils.label_mismatch(cluster_assignment_milp, z)
        if cluster_assignment_greedy is not None:
            greedy_mismatch = []
            for trails in range(len(cluster_assignment_greedy)):
                greedy_mismatch.append(Utils.label_mismatch(cluster_assignment_greedy[trails], z))
            result_df['label_mismatch_greedy'] = np.mean(greedy_mismatch)
        if 'MSE' in folder_name:
            result_df['mse_milp'] = mean_squared_error(y, y_pred)
            result_df['mse_refit_milp_assignment'] = mean_squared_error(y, y_pred)
            result_df['mse_refit_ground_truth_assignment'] = mean_squared_error(y, y_pred_refit_ground_truth)
            result_df['time_refit_ground_truth_assignment'] = refit_time
        else:
            result_df['mse_milp'] = mean_squared_error(y, y_pred)
            result_df['mse_refit_milp_assignment'] = mean_squared_error(y, y_pred)
            result_df['mse_refit_ground_truth_assignment'] = mean_squared_error(y, y_pred_refit_ground_truth)
            result_df['time_refit_ground_truth_assignment'] = refit_time

        if optimal_folder_path is not None:
            result_df['optimal_mse_difference_milp'] = np.abs(float(evaluation_results['mse_milp']) - float(optimal_evaluation_results['mse_milp'])).round(4)
            result_df['optimal_mse_difference_refit_milp_assignment'] = np.abs(float(evaluation_results['mse_refit_milp_assignment']) - float(optimal_evaluation_results['mse_refit_milp_assignment']))
            result_df['optimal_mse_difference_greedy'] = np.abs(float(evaluation_results['mse_greedy']) - float(optimal_evaluation_results['mse_refit_milp_assignment']))
            result_df['optimal_mse_difference_refit_ground_truth_assignment'] = np.abs(float(evaluation_results['mse_refit_ground_truth_assignment']) - float(optimal_evaluation_results['mse_refit_ground_truth_assignment']))
            if 'mse_CLR' in evaluation_results.keys():
                result_df['optimal_mse_difference_CLR'] = np.abs(float(evaluation_results['mse_CLR']) - float(optimal_evaluation_results['mse_milp'])).round(4)
            result_df['optimal_weight_difference_milp'] = Utils.weight_mismatch(regression_weight_milp, regression_weight_optimal, cluster_assignment_milp, optimal_cluster_assignment, check_unique=False)
            result_df['optimal_weight_difference_refit_milp_assignment'] = Utils.weight_mismatch(regression_weight_milp, regression_weight_optimal, cluster_assignment_milp, optimal_cluster_assignment, check_unique=False)
            result_df['optimal_weight_difference_refit_ground_truth_assignment'] = Utils.weight_mismatch(regression_weight_ground_truth, regression_weight_optimal, z, optimal_cluster_assignment, check_unique=False)
            result_df['optimal_cluster_mismatch_milp'] = Utils.label_mismatch(cluster_assignment_milp, optimal_cluster_assignment) / X.shape[0]
            result_df['optimal_cluster_mismatch_refit_milp_assignment'] = Utils.label_mismatch(cluster_assignment_milp, optimal_cluster_assignment) / X.shape[0]
            result_df['optimal_cluster_mismatch_refit_ground_truth_assignment'] = Utils.label_mismatch(z, optimal_cluster_assignment) / X.shape[0]
            result_df['optimal_cluster_mismatch_ground_truth'] = Utils.label_mismatch(z, optimal_cluster_assignment) / X.shape[0]
        return result_df
    
    @staticmethod
    def standard_error(data):
        return sem(data) 