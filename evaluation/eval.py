from synthetic_data.data import SyntheticData
from real_data.real import RealData
import gc
from scripts.Greedy_Codes.model import *
from scripts.Greedy_Codes.utils import *
from evaluation.utils import Utils
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

from scripts.model import LpcNsMip, LpcNsQbpo, GobalOpt, LinearPredictiveClustering
import numpy as np
import seaborn as sns
import pandas as pd
import os
import sys
import shutil
from time import time
import json
import inspect
from scipy.stats import sem


import matplotlib.pyplot as plt
from matplotlib import ticker
class Evaluation():
    '''
    Class for the evaluation framework of the model with several evaluation metrics and visualizations function for given data and model

    Attributes:
    data: SyntheticData | (future: RealData)
        The synthetic data object
    model: LinearPredictiveClustering
        The model object

    '''
    
    def __init__(self, data: SyntheticData|RealData, model:LinearPredictiveClustering = None, 
                 verbose:bool = True, random_state = 42, **kwargs):
        
        self.data = data
        self.model = model
        self.random = np.random.RandomState(random_state)
        random_state_greedy = self.random.randint(0, 1000, 20) # random initialization for the greedy model
        self.greedy = [SupervisedClustering(K=self.data.K, f=self.data.D, max_iter = 100, gmm=True, 
                                             random_state = i) for i in random_state_greedy] # initialize 20 greedy models for the baseline

        # initialize the model with the data
        if self.model != None:
            self.model.X = self.data.X[self.data.train, :]
            self.model.y = self.data.y[self.data.train]
            self.model.K = self.data.K
            self.model.n = len(self.data.train)
            self.model.p = self.data.D
            self.model.true_cluster_assignments = self.data.z[self.data.train]
            self.model.verbose = verbose

            self.model.random_state = self.data.random.get_state() # ensure the random state is consistent across the model and the data
            self.model.noise_std = self.data.noise_std

        # placeholder for the evaluation results
        self.regression_weights = {}
        self.cluster_assignments = {}
        self.evaluation_results = {}

        #ground truth cluster assignments
        #if isinstance(self.data, SyntheticData):
        self.regression_weights['ground_truth'] = [[self.data.cluster_params[k]['bias'][0]] + [i for i in self.data.cluster_params[k]['weights']] for k in range(self.data.K)]
        self.cluster_assignments['ground_truth'] = self.data.z[self.data.train]

        self.output_dir = f'Results/{self.data.name}/{self.data.N}_{self.data.K}_{self.data.D}_{self.data.noise_std}_{self.data.outlier_ratio}'
        if os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.verbose = verbose

        if 'writer' in kwargs:
            assert hasattr(kwargs['writer'], 'write'), 'The writer object must have a write method'
            self.writer = kwargs['writer']
        else:
            self.writer = sys.stderr
    
    def _print(self, text: str, **kwargs):
        '''
        an helper function to print and store the intermediate log if verbose is set to True or the writer is provided
        '''
        if self.verbose:
            print(text, file=self.writer,**kwargs)

    def fit(self, use_milp = True, whiten = True, use_greedy = False, refit = True, CLR = False, error = 'MSE', **kwargs):
        '''
        Applying model with the data and evaluate the model with the evaluation metrics

        Parameters:
        - use_milp: bool
            Whether to use the MILP model to compute the cluster assignments and regression weights
        - use_pbo: bool
            Whether to use the PBO model to compute the cluster assignments and regression weights
        - refit: bool
            Whether to refit the model with the cluster assignments from the MILP model
        - kwargs: dict
            The optional arguments to be passed to the model object to adjust the model parameters
        '''
        # update the model parameters based on the kwargs
        list_of_atributes = inspect.signature(self.model.__init__).parameters
        list_of_baseline_atributes = inspect.signature(self.greedy[0].__init__).parameters
        for key, value in kwargs.items():
            if key in list_of_atributes:
                setattr(self.model, key, value)
            if key in list_of_baseline_atributes:
                for i in self.greedy:
                    setattr(i, key, value)

        # fitting the model
        start_time = time()
        self.model.whiten = whiten
        self.model.loss = error
        self.model.prepare_augmented_data()
        if use_milp:
            self._print('Fitting the MIQP model...')
            self.model.perform_gurobi_miqp()
            self.model.compute_cluster_weights()
            milp_time = time() - start_time
            self._print(f'Fitting the model took {milp_time} seconds')
            self.evaluation_results['time_milp'] = milp_time
            self.regression_weights['milp'] = self.model.w_star_list
            self.cluster_assignments['milp'] = self.model.cluster_assignments

            self.evaluation_results['A*'] = self.model.A
            self.evaluation_results['A_k'] = self.model.A_k
    

        # fitting the baseline model from Arivith Work
        if use_greedy:
            for i in self.greedy:
                i.set_supervised_loss(LinearRegression(regularize_param = 2))
                i.set_assignment(ArbitraryAssign()) 
        
            self._print('Fitting the baseline greedy model...')
            #converting X and y to a pandas dataframe that can be used by the baseline model
            X = self.data.X[self.data.train, :]
            X_whiten = Utils.prepare_augmented_data(X, whiten)
            self.baseline_data = pd.DataFrame(X_whiten, columns = [f'x_{i}' for i in range(self.data.D)])
            self.baseline_data['y'] = self.data.y[self.data.train]

            self.cluster_assignments['greedy'] = []
            self.regression_weights['greedy'] = []
            self.evaluation_results['time_greedy'] = []
            
            for index, i in enumerate(self.greedy):
                # repeat the greedy model 10 times to get the average results
                start_time = time()
                i.fit(self.baseline_data)
                end_time = time() - start_time
                self._print(f'Fitting the baseline greedy model {index} took {end_time} seconds')
                try:
                    self.regression_weights['greedy'].append(Utils.perform_refit_sklearn_ridge_regression(X, self.data.y[self.data.train], self.model.K,
                                                                                                          i.data.model.values - 1, loss = 'MSE',
                                                                                                            lambda_reg=self.model.lambda_reg))
                    self.evaluation_results['time_greedy'].append(end_time)
                    self.cluster_assignments['greedy'].append(i.data.model.values - 1)
                except:
                    print(f'Error in the {index} greedy model regression weights')
                    self.regression_weights['greedy'].append(self.regression_weights['greedy'][-1])
                    self.evaluation_results['time_greedy'].append(end_time)
                    self.cluster_assignments['greedy'].append(self.cluster_assignments['greedy'][-1])
                    
        # refit the model with the cluster assignments from the MILP or PBO model
        if refit and (use_milp):
            start_time = time()
            self._print('Refitting the model...')
            self.model.perform_refit_sklearn_ridge_regression()
            refit_time = time() - start_time
            self._print(f'Refitting the model took {refit_time} seconds') 
            self.evaluation_results['time_refit_milp_assignment'] = refit_time + milp_time
            self.regression_weights['refit_milp_assignment'] = self.model.w_star_refit
            self.cluster_assignments['refit_milp_assignment'] = self.model.cluster_assignments
        
        if CLR:
            start_time = time()
            y_pred = np.zeros((self.data.X.shape[0], 1))
            kmeans = KMeans(n_clusters=self.data.K, random_state=0).fit(self.data.X)
            z_Pred = kmeans.labels_
            for i in range(2):
                x_k = self.data.X[z_Pred == i]
                y_k = self.data.y[z_Pred == i]
                model =  Ridge(alpha=self.model.lambda_reg)
                model.fit(x_k, y_k)
                y_pred[z_Pred == i] = model.predict(x_k).reshape(-1, 1)
            mse, r2 = Utils.mse_r2(self.data.y, y_pred)
            print(f'CLR took {time() - start_time} seconds')
            self.evaluation_results['mse_CLR'] = mse
            print(f'MSE: {mse}')
            self.evaluation_results['r2_CLR'] = r2
            self.evaluation_results['rand_score_CLR'] = rand_score(z_Pred, self.data.z)
            self.evaluation_results['label_mismatch_CLR'] = Utils.label_mismatch(z_Pred, self.data.z)/len(self.data.z)
            self.evaluation_results['time_CLR'] = time() - start_time
        self.K = self.model.K

    def _predict(self, X, regression_weights_by_k, cluster_assignments):
        '''
        A helper function to predict the y values with the given regression weights and cluster assignments

        Parameters:
        regression_weights_by_k: list
            The list of regression weights for each cluster
        cluster_assignments: np.array
            The cluster assignments for each data point
        '''

        assert np.shape(X)[0] == len(cluster_assignments), 'The number of data points must be the same as the number of cluster assignments'
        y_pred = np.zeros(len(cluster_assignments))
        for k in np.unique(cluster_assignments): # ingore cluster with no data points
            cluster_indices = np.where(cluster_assignments == k)[0]
            y_pred[cluster_indices] = regression_weights_by_k[k][0] + np.dot(X[cluster_indices], regression_weights_by_k[k][1:])
        return y_pred
    
    def nearest_neighbor_assignment(self, cluster_assignments, new_data):
        '''
        A helper function to assign the cluster assignments as the nearest neighbor cluster assignment for the new data

        Parameters:
        cluster_assignments: np.array
            The cluster assignments for each existing data point

        new_data: np.array
            The new data points to be assigned to the nearest neighbor cluster
        '''
        data = np.column_stack((self.data.X[self.data.train, :], self.data.y[self.data.train]))
        knn = NearestNeighbors(n_neighbors=1).fit(data, cluster_assignments)
        _, indices  = knn.kneighbors(new_data)
        cluster_label = [cluster_assignments[i] for i in indices]
        return np.array(cluster_label).flatten()

    def evaluate(self):
        '''
        Evaluate the model with the several evaluation metrics
        '''
        # MSE for the MILP regression and assignment
        # reset existing evaluation results except for the time
        self.evaluation_results = {key: value for key, value in self.evaluation_results.items() if 'time' in key or 'CLR' in key}
        self.y_pred = {}
        
        for model in ['refit_ground_truth_assignment','milp', 'refit_milp_assignment', 'greedy', 'ground_truth']:

            if model == 'refit_ground_truth_assignment':
                self.regression_weights[model] = self.model.perform_refit_sklearn_ridge_regression(ground_truth = True)
                self.cluster_assignments[model] = self.data.z[self.data.train]

            if model == 'ground_truth':
                if isinstance(self.data, SyntheticData):
                    self.evaluation_results[f'mse_{model}'] = self.data.ground_truth_mse
                    self.evaluation_results[f'r2_{model}'] = self.data.ground_truth_r2
                else:
                    self.evaluation_results[f'mse_{model}'] = np.nan
                    self.evaluation_results[f'r2_{model}'] = np.nan

                self.evaluation_results[f'weight_mismatch_{model}'] = 0
                self.evaluation_results[f'rand_score_{model}'] = 1
                self.evaluation_results[f'label_mismatch_{model}'] = 0

            elif model == 'greedy' and 'time_greedy' in self.evaluation_results:
                time_lst = self.evaluation_results['time_greedy']
                mse_lst = []
                r2_lst = []
                rand_score_lst = []
                label_mismatch_lst = []
                weight_mismatch_lst = []
                weight_mismatch_refit_lst = []
                X = self.data.X[self.data.train, :]
                for i in range(len(self.regression_weights[model])):
                    y_pred = np.zeros(len(self.data.train))
                    for k in np.unique(self.cluster_assignments[model][i]):
                        cluster_indices = np.where(self.cluster_assignments[model][i] == k)[0]
                        y_pred[cluster_indices] = X[cluster_indices] @ self.regression_weights[model][i][k][1:] + self.regression_weights[model][i][k][0]
                    self.y_pred[model] = y_pred
                    mse, r2 = Utils.mse_r2(y_pred, self.data.y[self.data.train], error = self.model.loss)
                    mse_lst.append(mse)
                    r2_lst.append(r2)
                    rand_score_lst.append(rand_score(self.cluster_assignments[model][i], self.data.z[self.data.train]))
                    label_mismatch_lst.append(Utils.label_mismatch(self.cluster_assignments[model][i], self.data.z[self.data.train])/len(self.data.train))
                    weight_mismatch_lst.append(Utils.weight_mismatch(self.regression_weights[model][i], self.regression_weights['ground_truth'], self.cluster_assignments[model][i], self.data.z[self.data.train]))
                    weight_mismatch_refit_lst.append(Utils.weight_mismatch(self.regression_weights[model][i], self.regression_weights['refit_ground_truth_assignment'], self.cluster_assignments[model][i], self.data.z[self.data.train]))
                
                self.evaluation_results[f'mse_{model}'] = np.mean(mse_lst)
                self.evaluation_results[f'r2_{model}'] = np.mean(r2_lst)
                self.evaluation_results[f'weight_mismatch_{model}'] = np.mean(weight_mismatch_lst)
                self.evaluation_results[f'refit-weight_mismatch_{model}'] = np.mean(weight_mismatch_refit_lst)
                self.evaluation_results[f'rand_score_{model}'] = np.mean(rand_score_lst)
                self.evaluation_results[f'label_mismatch_{model}'] = np.mean(label_mismatch_lst)
                self.evaluation_results[f'time_{model}'] = np.mean(time_lst)
                
                # calculate the sem for the evaluation metrics
                self.evaluation_results[f'mse_{model}_sem'] = sem(mse_lst)
                self.evaluation_results[f'r2_{model}_sem'] = sem(r2_lst)
                self.evaluation_results[f'weight_mismatch_{model}_sem'] = sem(weight_mismatch_lst)
                self.evaluation_results[f'refit-weight_mismatch_{model}_sem'] = sem(weight_mismatch_refit_lst)
                self.evaluation_results[f'rand_score_{model}_sem'] = sem(rand_score_lst)
                self.evaluation_results[f'label_mismatch_{model}_sem'] = sem(label_mismatch_lst)


            elif model in self.regression_weights:
                y_pred = self._predict(self.data.X[self.data.train, :], self.regression_weights[model], self.cluster_assignments[model])
                self.y_pred[model] = y_pred
                mse, r2 = Utils.mse_r2(y_pred, self.data.y[self.data.train],error = self.model.loss)
                self.evaluation_results[f'mse_{model}'] = mse
                self.evaluation_results[f'r2_{model}'] = r2
                self.evaluation_results[f'weight_mismatch_{model}'] = Utils.weight_mismatch(self.regression_weights[model], self.regression_weights['ground_truth'], self.cluster_assignments[model], self.data.z[self.data.train])
                self.evaluation_results[f'refit-weight_mismatch_{model}'] = Utils.weight_mismatch(self.regression_weights[model], self.regression_weights['refit_ground_truth_assignment'], self.cluster_assignments[model], self.data.z[self.data.train])
                self.evaluation_results[f'rand_score_{model}'] = rand_score(self.cluster_assignments[model], self.data.z[self.data.train])
                self.evaluation_results[f'label_mismatch_{model}'] = Utils.label_mismatch(self.cluster_assignments[model], self.data.z[self.data.train])/len(self.data.train)
            
            else:
                if self.verbose:
                    print(f'{model} model is not fitted')
                self.evaluation_results[f'mse_{model}'] = np.nan
                self.evaluation_results[f'r2_{model}'] = np.nan

        #MSE for baseline model with sklearn ridge regression on the entire data without clustering
        self.model.perform_sklearn_ridge_regression()
        self.evaluation_results['mse_baseline_sklearn'] = self.model.mse_sklearn
        self.evaluation_results['r2_baseline_sklearn'] = self.model.r2_sklearn
        #self.evaluation_results['rand_score_baseline_sklearn'] = rand_score([0]*len(self.data.train), self.data.z[self.data.train])

        # # MSE for the ground truth regression line that generated the data
        if isinstance(self.data, SyntheticData):
            self.evaluation_results['mse_ground_truth'] = self.data.ground_truth_mse
            self.evaluation_results['r2_ground_truth'] = self.data.ground_truth_r2


        if self.data.val is not None:
            val_data = np.column_stack((self.data.X[self.data.val, :], self.data.y[self.data.val]))
            # if the dataset contains a validation set, evaluate the model with the validation set
            # 1. obtain the cluster assignments for the validation set
            for model in ['milp', 'refit_milp_assignment']:
                if model in self.cluster_assignments:
                    cluster_assignments_val = self.nearest_neighbor_assignment(self.cluster_assignments[model], val_data)
                    # 2. predict the y values for the validation set, calculate the MSE, R2, and cluster purity
                    y_pred = self._predict(self.data.X[self.data.val, :], self.regression_weights[model], cluster_assignments_val)
                    mse, r2 = Utils.mse_r2(y_pred, self.data.y[self.data.val], error = self.model.loss)
                    self.evaluation_results[f'mse_{model}_val'] = mse
                    self.evaluation_results[f'r2_{model}_val'] = r2
                    label_mismatch = Utils.label_mismatch(cluster_assignments_val,self.data.z[self.data.val])/len(self.data.val)
                    self.evaluation_results[f'label_mismatch_{model}_val'] = label_mismatch

            model = 'greedy'
            if model in self.cluster_assignments:
                label_mismatch_lst = []
                mse_lst = []
                r2_lst = []

                for i in range(len(self.regression_weights[model])):
                    cluster_assignments_val = self.nearest_neighbor_assignment(self.cluster_assignments[model][i], val_data)
                    y_pred = self._predict(self.data.X[self.data.val, :], self.regression_weights[model][i], cluster_assignments_val)
                    mse, r2 = Utils.mse_r2(y_pred, self.data.y[self.data.val], error = self.model.loss)
                    label_mismatch = Utils.label_mismatch(cluster_assignments_val,self.data.z[self.data.val])/len(self.data.val)
                    mse_lst.append(mse)
                    r2_lst.append(r2)
                    label_mismatch_lst.append(label_mismatch)
                self.evaluation_results[f'mse_{model}_val'] = np.mean(mse_lst)
                self.evaluation_results[f'label_mismatch_{model}_val'] = np.mean(label_mismatch_lst)
                self.evaluation_results[f'mse_{model}_val_sem'] = sem(mse_lst)
                self.evaluation_results[f'label_mismatch_{model}_val_sem'] = sem(label_mismatch_lst)
                self.evaluation_results[f'r2_{model}_val'] = np.mean(r2_lst)
                self.evaluation_results[f'r2_{model}_val_sem'] = sem(r2_lst)

            # ground truth label to assign cluster label
            cluster_assignments_val = self.nearest_neighbor_assignment(self.cluster_assignments['ground_truth'], val_data)
            for model in ['refit_ground_truth_assignment', 'ground_truth']:
                y_pred = self._predict(self.data.X[self.data.val, :], self.regression_weights[model], cluster_assignments_val)
                mse, r2 = Utils.mse_r2(y_pred, self.data.y[self.data.val], error = self.model.loss)
                label_mismatch = Utils.label_mismatch(cluster_assignments_val, self.data.z[self.data.val])/len(self.data.val)
                self.evaluation_results[f'mse_{model}_val'] = mse
                self.evaluation_results[f'r2_{model}_val'] = r2
                self.evaluation_results[f'label_mismatch_{model}_val'] = label_mismatch 

            #using basline ridge regression
            y_pred = self.model.ridge_model.predict(self.data.X[self.data.val, :])
            cluster_assignments_val = [0]*len(self.data.val)
            mse, r2 = Utils.mse_r2(y_pred, self.data.y[self.data.val], error = self.model.loss)
            self.evaluation_results['mse_baseline_sklearn_val'] = mse
            self.evaluation_results['r2_baseline_sklearn_val'] = r2
            #self.evaluation_results['label_mismatch_baseline_sklearn_val'] = 0

            


    def print_evaluation_results(self):
        '''
        Print the evaluation results in a visual friendly format
        '''
        tem = pd.DataFrame(self.evaluation_results, index=['value']).transpose()
        tem.reset_index(inplace=True)
        tem.columns = ['metric', 'value']
        tem['model'] = tem['metric'].apply(lambda x: ' '.join([i for i in x.split('_')[1:] if i != 'val' and i != 'mismatch' and i != 'score']).strip().title())
        tem['metric'] = tem['metric'].apply(lambda x: x.split('_')[0] + ' ' + [i if i == 'val' else '' for i in [x.split('_')[-1]]][0])
        tem['metric'] = tem['metric'].apply(lambda x: x + ' mismatch' if 'label' in x or 'weight' in x else x)
        #drop row that contains SEM in model
        tem = tem[~tem['model'].str.contains('Sem')]
        tem_val = tem[tem['metric'].str.contains('val')]
        tem = tem[~tem['metric'].str.contains('val')]
        tem = tem.pivot(index='metric', columns='model', values='value')
        tem.loc['label  mismatch','Greedy'] = str(tem.loc['label  mismatch','Greedy']) + '±' + str(self.evaluation_results['label_mismatch_greedy_sem'])
        tem.loc['mse ','Greedy'] = str(tem.loc['mse ','Greedy']) + '±' + str(self.evaluation_results['mse_greedy_sem'])
        tem.loc['r2 ','Greedy'] = str(tem.loc['r2 ','Greedy']) + '±' + str(self.evaluation_results['r2_greedy_sem'])
        tem.loc['rand ','Greedy'] = str(tem.loc['rand ','Greedy']) + '±' + str(self.evaluation_results['rand_score_greedy_sem'])
        tem.loc['refit-weight  mismatch','Greedy'] = str(tem.loc['refit-weight  mismatch','Greedy']) + '±' + str(self.evaluation_results['refit-weight_mismatch_greedy_sem'])
        tem.loc['weight  mismatch','Greedy'] = str(tem.loc['weight  mismatch','Greedy']) + '±' + str(self.evaluation_results['weight_mismatch_greedy_sem'])

        tem_val = tem_val.pivot(index='metric', columns='model', values='value')
        return tem, tem_val
            


    def plot(self, model:str = 'milp', show_cluster = True, show_regresion = True, fig = None, axes = None, validation = False, **kwargs):
        '''
        Plot data, cluster assignments, and regression lines with controls for the visualization

        Parameters:
        model: str
            The model to be plotted, default is 'milp'

        show_cluster: bool
            Whether to show the cluster assignments

        show_regresion: bool
            Whether to show the regression lines
        '''
        title = ''
        ls = {'milp': '-', 'greedy': '--', 'ground_truth': '-.'}
        if model in self.cluster_assignments:
            if show_cluster:
                cluster_assignments = self.cluster_assignments[model]

                if model == 'greedy':   
                    cluster_assignments = cluster_assignments[0]

                fig, axes = Evaluation.plot_data_with_cluster_label(self.data, label = cluster_assignments, 
                                                                    fig = fig, axes = axes,
                                                                    ls = ls[model], validate = False, **kwargs)
                title += f'{model} cluster assignments'

                if validation:
                    title += 'Validation '
                    val_data = np.column_stack((self.data.X[self.data.val, :], self.data.y[self.data.val]))
                    cluster_assignments = self.nearest_neighbor_assignment(self.cluster_assignments[model], val_data)
                    print(cluster_assignments)
                    fig, axes = Evaluation.plot_data_with_cluster_label(self.data, label = cluster_assignments,
                                                                        fig = fig, axes = axes, ls = ls[model], 
                                                                        validate = True, **kwargs)

            if show_regresion:      
                if model == 'greedy':
                    regression_weights = self.regression_weights[model][0]
                else:
                    regression_weights = self.regression_weights[model]
                fig, axes = Evaluation.plot_regression_results(self.data, regression_weights= regression_weights, 
                                                            fig = fig, axes = axes, label = model,ls = ls[model], alpha = 0.5, **kwargs)
                title += f' and {model} regression line'

                if model == 'milp':
                    fig, axes = Evaluation.plot_regression_results(self.data, regression_weights=self.regression_weights['refit_milp_assignment'],
                                                                fig = fig, axes = axes, label = 'refit', ls = '--', alpha = 0.5, **kwargs)
                    title += ' with refit'
                if model == 'ground_truth':
                    if 'refit_ground_truth_assignment' not in self.regression_weights:
                        self.regression_weights['refit_ground_truth_assignment'] = self.model.perform_refit_sklearn_ridge_regression(ground_truth = True) 
                    fig, axes = Evaluation.plot_regression_results(self.data, regression_weights=self.regression_weights['refit_ground_truth_assignment'],
                                                                fig = fig, axes = axes, label = 'refit', ls = '--', alpha = 0.5, **kwargs)
                    title += ' with refit'

            fig.suptitle(title.title(), fontsize=10)
        return fig, axes
    
    @staticmethod
    def _save_result(results, output_dir):
        '''
        Save the evaluation results to a file
        '''
        if type(results) == plt.Figure:
            results.savefig(output_dir)
        else:
            assert type(results) == dict, 'The results must be a dictionary'
            assert output_dir.endswith('.json'), 'The output directory must be a json file'

            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
                
            with open(output_dir, 'w') as f:
                json.dump(results, f, cls=NpEncoder)

    @staticmethod
    def plot_data_with_cluster_label(data, label = None, fig = None, axes = None, validate = False, **kwargs):
        '''
        Return a scatter plot of the data, if label is provided, the data points will be colored by the label

        Parameters:
        data: SyntheticData | (future: RealData)
            The data object

        label: np.array|None
            The label to be used for the scatter plot
        '''
        if label is None:
            color = 'gray'
        else:
            color = [sns.color_palette()[i] for i in label]

        if validate:
            X = data.X[data.val,:]
            y = data.y[data.val]
            marker = 'x'
        
        else:
            X = data.X[data.train,:]
            y = data.y[data.train]
            marker = 'o'

        if data.D == 1:
            # handle 1D data
            # optional: pass in fig and axes to plot on the same figure
            if fig is None and axes is None:
                fig = plt.figure()
                axes = fig.add_subplot(111)

            axes.scatter(X, y, c = color, marker=marker, **kwargs)
            axes.set_xlabel('X')
            axes.set_ylabel('y')
            if not validate:
                axes.text(np.min(X), np.min(y), f'N = {len(y)} \n K = {data.K}\n D = {data.D} \n noise_std = {data.noise_std}\n outlier_ratio = {data.outlier_ratio}', fontsize=10, ha='left') 
        else:
            # handle multi-dimensional data
            if fig is None and axes is None:
                fig, axes = plt.subplots(1, data.D, figsize=(data.D*5, 5))
                
            for d in range(data.D):
                axes[d].scatter(X[:, d], y, marker=marker,  c = [sns.color_palette()[i] for i in label])
                axes[d].set_title(f'Feature {d+1}')
                axes[d].set_xlabel(f'X{d+1}')
                if d == 0:
                    axes[d].set_ylabel('y')
            
            if not validate:
                axes[d].text(np.min(X[:,d]), np.min(y), f'N = {len(y)} \n K = {data.K}\n D = {data.D} \n noise_std = {data.noise_std}\n outlier_ratio = {data.outlier_ratio}', fontsize=10, ha='left')

        return fig, axes

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
    
    def print_model_results(self):
        '''
        A helper function to better visual the regression weights for each cluster from model results
        '''
        print("Cluster assignments: ", self.cluster_assignments['milp'])
        if isinstance(self.data, SyntheticData):
            for k in range(len(self.regression_weights['milp'])):
                print('-----------------------------------')
                print(f'''Regression weights for cluster {k}: y = {" + ".join([str(round(weight, 4)) + f"x_{i}" for i, weight in enumerate(self.regression_weights["milp"][k][1:])])} + {round(self.regression_weights["milp"][k][0], 4)}''')
                print(f"Regression weights for cluster {k} after refit: y = {' + '.join([str(round(weight, 4)) + f'x_{i}' for i, weight in enumerate(self.regression_weights['refit_milp_assignment'][k])][1:])} + {round(self.regression_weights['refit_milp_assignment'][k][0], 4)}")
            return None
        
        if isinstance(self.data, RealData):
            df = pd.DataFrame()
            for k in range(len(self.regression_weights['milp'])):
            #   print('-----------------------------------')
            #   print(f'''Regression weights for cluster {k}: y = {" + ".join([str(round(weight, 4)) + f"x_{i}" for i, weight in enumerate(self.regression_weights["milp"][k][1:])])} + {round(self.regression_weights["milp"][k][0], 4)}''')
            #   print(f"Regression weights for cluster {k} after refit: y = {' + '.join([str(round(weight, 4)) + ' ' + self.data.feature_name[i] for i, weight in enumerate(self.regression_weights['refit_milp_assignment'][k])][1:])} + {round(self.regression_weights['refit_milp_assignment'][k][0], 4)}")
                df = pd.concat([df, pd.DataFrame(self.regression_weights['milp'][k], columns = [f'cluster_{k}'])], axis = 1)
            
            index = ['bias'] + self.data.feature_name
            df.index = index

            return df
    
    @staticmethod
    def evaluate_model(data_template: SyntheticData, model: LinearPredictiveClustering, 
                              parameter: str, parameter_value: float, dataset = 0, whiten = False,
                              as_model_parameter:bool = False, lambda_values = None, validate = False,
                              verbose = True, random_state = 0, save_local = False, show_plot = True,
                              use_greedy_only = False, CLR = False,
                              file_dir = 'Results', loss = 'MSE', data_dir = None, **kwargs) -> tuple:

        '''
        A function to evaluate the model with a single value of the parameter and lambda for parallel computing
        
        Parameters:
        - data_template: SyntheticData | (future: RealData)
        - model: LinearPredictiveClustering
            The model object to be evaluated
        - parameter: str
            The parameter to be varied, should be within the controllable parameters of the data object
        - as_model_parameter: bool
            Whether the parameter is a model parameter, if True, the parameter will be varied in the model object, else in the data object
        - values: float
            The value to be evaluated
        - lambda_values: float
            The regularization parameter for the linear regression
        - verbose: bool
            Whether to print the intermediate log, if false, only minimum information will be printed
        - plot_results: bool
            Whether to plot the regression line during the evaluation
        - save_local: bool
            Whether to save the evaluation results locally
        - show_plot: bool
            Whether to show the plot of the evaluation results
        - file_dir: str
            The directory to store the evaluation results
        - loss: str
            The loss function to be used for the evaluation, default is 'MSE'

        '''
        print(f'==================== Evaluating with {parameter} = {parameter_value} in Dataset {dataset} with random state = {random_state} ====================')

        list_of_parameters = inspect.signature(data_template.generate_data).parameters

        # default data parameters
        data_args = {'N': 120, 'K': 2, 'D': 1, 'noise_std': 2, 'outlier_ratio': 0.0, 'validation': validate}
        if verbose and validate:
            print('ODS is enabled')
        model_class = model.__class__

        # update the input parameters based on the kwargs
        for key, value in kwargs.items():
            if key in list_of_parameters:
                data_args[key] = value

        # obtain the model and data parameters
        if as_model_parameter:
            model_args = {parameter: parameter_value}
        elif parameter in list_of_parameters:
            data_args[parameter] = parameter_value
            model_args = {}
        else:
            model_args = {}

        # 1. Initialize the data and model objects
        data = SyntheticData(random_state = random_state)
        use_existing_data = False

        if data_dir is not None:
            data.load_data(data_dir)
            use_existing_data = True

        # update the data object with the data template
        for data_template_key, data_template_value in data_template.__dict__.items():
            if data_template_key in ['cluster_size_generator', 'cluster_label_generator',
                                        'regression_params_generator', 
                                        'regression_target_generator', 
                                        'outlier_generator']:
                setattr(data, data_template_key, data_template_value)
        model = model_class()
        model_args['lambda_reg'] = lambda_values
        model.loss = loss

        name = []
        for item in [data.cluster_label_generator, data.cluster_size_generator, 
                     data.regression_params_generator, data.regression_target_generator,data.outlier_generator]:
            if item is not None:
                name.append(item.__class__.__name__)
        name = '_'.join(name)

        data_args['use_exist'] = use_existing_data
        # 2. Generate the synthetic data
        data.generate_data(**data_args)

        # 3. Initialize the evaluation object
        evaluation = Evaluation(data, model, verbose=verbose, random_state=random_state)
        for greedy in evaluation.greedy:
             greedy.set_supervised_loss(LinearRegression(regularize_param = lambda_values))
        # 4. Fit the model with the synthetic data to obtain the cluster assignments and regression coefficients
        if use_greedy_only:
            evaluation.fit(refit = False, use_milp=False, use_greedy=True, 
                           whiten=whiten, CLR=CLR, **model_args)
        elif loss == 'QPBO':
            evaluation.fit(refit = True, use_greedy=True, whiten=whiten,
                            error=loss, CLR=CLR, **model_args)
        else:
            evaluation.fit(refit = True, use_greedy=False, whiten=whiten, 
                           error=loss, CLR=CLR, **model_args)

        # 5. Evaluate the model with the evaluation metrics
        evaluation.evaluate()

        if verbose:
            evaluation.print_model_results()
            print(evaluation.evaluation_results)

        if save_local:
            # used in local machine to store the results
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)

            dir = f'{file_dir}/{parameter}/' + '_'.join([f'{key}={value}' for key, value in data_args.items() if key != parameter]) + f'_{name}' + f'/{parameter}={parameter_value}/random_state={dataset}/lambda={random_state}'
            
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                shutil.rmtree(dir)
                os.makedirs(dir)
            
            if show_plot:
            # 6. Visualize the results and store them in the directory
                img_dir = f'{file_dir}_img/{parameter}/' + '_'.join([f'{key}={value}' for key, value in data_args.items() if key != parameter]) + f'_{name}' + f'/{parameter}={parameter_value}/random_state={dataset}/lambda={random_state}'
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                else:
                    shutil.rmtree(img_dir)
                    os.makedirs(img_dir)

                evaluation.plot(model = 'milp', show_cluster = True, show_regresion = True)
                plt.savefig(img_dir + '/milp.png')
                plt.close()
                evaluation.plot(model = 'ground_truth', show_cluster = True, show_regresion = True)
                plt.savefig(img_dir + '/ground_truth.png')
                plt.close()
                evaluation.plot(model = 'greedy', show_cluster = True, show_regresion = True)
                plt.savefig(img_dir + '/greedy.png')
                plt.close()

            
            evaluation.evaluation_results['random_state'] = random_state
            json.dump(evaluation.evaluation_results, open(dir+"/evaluation_results.json", 'w'), cls=NpEncoder)
            json.dump(evaluation.regression_weights, open(dir+"/regression_weights.json", 'w'), cls=NpEncoder)
            json.dump(evaluation.cluster_assignments, open(dir+"/cluster_assignments.json", 'w'), cls=NpEncoder)
            data.save_data(dir+"/data.json")
            return None
        # if not save_local, return the evaluation results (used in compute canada to store the results within the wrapper function)
        return evaluation.evaluation_results, evaluation.regression_weights, evaluation.cluster_assignments, data