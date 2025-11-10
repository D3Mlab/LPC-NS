
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker

class Visualization():

    #static variables
    label_mapper = {'lpc_ns_qpbo_ground_truth': 'Ground Truth',

                    'lpc_ns_mip_milp': 'lpc_ns_mip without Refit',
                    'lpc_ns_mip_refit_milp_assignment': 'LPC-NS-MIP',  
                    'lpc_ns_qpbo_milp': 'LPC-NS without Refit',
                    'lpc_ns_qpbo_refit_milp_assignment': 'LPC-NS-QPBO',
                    
                    'lpc_ns_qpbo_greedy': 'Greedy',
                    'lpc_ns_mip_greedy': 'Greedy',
                    'globalopt_greedy': 'Greedy',

                    'globalopt_milp': 'GlobalOpt'}

    color_mapper = {'lpc_ns_qpbo_ground_truth': 'grey',

                    'lpc_ns_mip_milp': 'green',
                    'lpc_ns_mip_refit_milp_assignment': 'green',  
                    'lpc_ns_qpbo_milp': 'red',
                    'lpc_ns_qpbo_refit_milp_assignment': 'red',
                    
                    'lpc_ns_qpbo_greedy': 'blue',
                    'lpc_ns_mip_greedy': 'blue',
                    'globalopt_greedy': 'blue',

                    'globalopt_milp': 'black'}
        
    marker_mapper = {'lpc_ns_qpbo_ground_truth': '*',

                    'lpc_ns_mip_milp': '.',
                    'lpc_ns_mip_refit_milp_assignment': 'o',  
                    'lpc_ns_qpbo_milp': '*',
                    'lpc_ns_qpbo_refit_milp_assignment': 'x',
                    
                    'lpc_ns_qpbo_greedy': 'h',
                    'lpc_ns_mip_greedy': 'h',
                    'globalopt_greedy': 'h',

                    'globalopt_milp': 's'}
    
    xlabel_mapper = {'N': 'Number of Samples', 'K': 'Number of Clusters', 'D': 'Dimension of X', 
                    'noise_std': 'sd of Gaussian Noise', 
                    'outlier_ratio': 'Proportion of Outliers (%)', 
                    'imbalanced_ratio': 'Ratio of Imbalanced Clusters',
                      'V': 'Difference Between Top Eigenvector',
                    'XTX': 'difference in Covariance Matrix'}
    
    metric_mapper = {'mse': 'SSE', 'weight_mismatch': 'Regression Coeffcients Difference',
                      'refit-weight_mismatch': 'Regression Difference', 
                     'label_mismatch': 'Cluster Mismatch(%)',
                       'mse_val': 'SSE (Out Of Distribution)', 
                     'label_mismatch_val': 
                     'Cluster Assignment Mismatch (Out Of Distribution)'}
    
    def __init__(self, evaluation_results = pd.DataFrame()):
        self.evaluation_results = evaluation_results

    @staticmethod
    def plot_evaluation_line_plot(parameter: str,
                                        lpc_ns_mip: pd.DataFrame,
                                        lpc_ns_qpbo: pd.DataFrame,
                                        globalopt: pd.DataFrame,
                                        x_axis = 'time',
                                        subset = [],
                                        list_of_parameters = None,
                                        legend_position = 1,
                                        legend = True):
        if x_axis == 'time':  
            fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        else:
            fig, ax = plt.subplots(1, 3, figsize = (15, 5))

        data_dict = {'lpc_ns_mip': lpc_ns_mip, 
                     'lpc_ns_qpbo': lpc_ns_qpbo, 
                     'globalopt': globalopt}
        
        if list_of_parameters is not None:
            if parameter == 'D':
                list_of_parameters = np.array(list_of_parameters).astype(int)
            else:
                list_of_parameters = np.array(list_of_parameters).round(2)
            common_parameters = list_of_parameters
        else:
            common_parameters = list(set(lpc_ns_mip[parameter].round(2).unique()).intersection(set(lpc_ns_qpbo[parameter].round(2).unique())))
        
        for key, df in data_dict.items():
            if parameter == 'D':
                data_dict[key][parameter] = data_dict[key][parameter].astype(int)
            else:
                data_dict[key][parameter] = data_dict[key][parameter].round(2)
            common_parameters = list(set(common_parameters).intersection(set(data_dict[key][parameter].unique())))
        for key, df in data_dict.items():
            data_dict[key] = data_dict[key][data_dict[key][parameter].isin(common_parameters)].reset_index(drop = True)

        sem_dict = {}
        for key, df in data_dict.items():
            result_df = df.copy()
            result_df_sem = df.copy()
            for i in range(result_df.shape[0]):
                for j in range(result_df.shape[1]):
                    if "±" in str(result_df.iloc[i, j]):
                        result_df.iloc[i, j] = float(result_df.iloc[i, j].split("±")[0])
                        result_df_sem.iloc[i, j] = float(result_df_sem.iloc[i, j].split("±")[1])
            result_df = result_df.astype(float)
            result_df_sem = result_df_sem.astype(float)
            sem_dict[key] = result_df_sem
            data_dict[key] = result_df

        for key, df in data_dict.items():
            result_df = df
            result_df_sem = sem_dict[key]
            if key == 'lpc_ns_mip':
                models = ['greedy', 'milp', 'refit_milp_assignment']
            if key == 'lpc_ns_qpbo':
                models = ['milp', 'refit_milp_assignment', 'greedy']
            if key == 'globalopt'or key == 'optimal_original':
                models = ['milp']

            for i, column in enumerate(models):
                tag = key + '_' + column
                if subset != [] and tag in subset:
                    if x_axis == 'time':
                        x = result_df[f'time_{column}']
                        x_err = result_df_sem[f'time_{column}']
                    else:
                        x = result_df[parameter] 
                        x_err = [0] * result_df.shape[0]

                    y = result_df['optimal_mse_difference_'+column]
                    if key == 'globalopt':
                        y = y + 1e-4 # to avoid log(0)

                    y_err = result_df_sem['optimal_mse_difference_'+column]
                    y_err = 1.96 * y_err              
                    ax[0].plot(x, y, label = Visualization.label_mapper[tag], 
                            markerfacecolor='none', markersize=20, ls = '--', alpha = 0.8, 
                            c = Visualization.color_mapper[tag], marker = Visualization.marker_mapper[tag])
                    for j in range(result_df.shape[0]):
                        ax[0].errorbar(x[j], y[j], xerr = x_err[j], yerr = y_err[j], marker = Visualization.marker_mapper[tag],
                            markerfacecolor='none', markersize=20, alpha = 0.8, c = Visualization.color_mapper[tag], capsize=5)
                    
                    if x_axis == 'time':
                        ax[1].plot(result_df[parameter], result_df['time_'+column], label = Visualization.label_mapper[tag],
                                markerfacecolor='none', markersize=20, ls = '--', alpha = 0.8, c = Visualization.color_mapper[tag], 
                                marker = Visualization.marker_mapper[tag])
                    else:
                        ax[1].plot(x, result_df['refit-weight_mismatch_'+column], label = Visualization.label_mapper[tag],
                                    markerfacecolor='none', markersize=20, ls = '--', alpha = 0.8, c = Visualization.color_mapper[tag],
                                    marker = Visualization.marker_mapper[tag])
                        y_err = result_df_sem['refit-weight_mismatch_'+column]
                        #as 95% confidence interval
                        y_err = 1.96 * y_err  
                        y = result_df['refit-weight_mismatch_'+column]
                        for j in range(result_df.shape[0]):
                            ax[1].errorbar(x[j], y[j],
                                                xerr = x_err[j], yerr = y_err[j], marker = Visualization.marker_mapper[tag],
                                markerfacecolor='none', markersize=20, alpha = 0.8, c = Visualization.color_mapper[tag], capsize=5)
                        
                        # label mismatch
                        ax[2].plot(x, result_df['label_mismatch_'+column], marker = Visualization.marker_mapper[tag], label = Visualization.label_mapper[tag],
                                markerfacecolor='none', markersize=20, ls = '--', alpha = 0.8, c = Visualization.color_mapper[tag])
                        y_err = result_df_sem['label_mismatch_'+column]
                       #as 95% confidence interval
                        y_err = 1.96 * y_err
                        y = result_df['label_mismatch_'+column]
                        for j in range(result_df.shape[0]):
                            ax[2].errorbar(x[j], result_df['label_mismatch_'+column][j],
                                            xerr = x_err[j], yerr = y_err[j], 
                                            marker = Visualization.marker_mapper[tag], 
                                            markerfacecolor='none', markersize=20, 
                                            alpha = 0.8, c = Visualization.color_mapper[tag], capsize=5)
        
        ax[0].set_ylabel('Diff From globalopt Obj', fontsize = 25)
        if x_axis == 'time': 
            ax[1].set_ylabel('', fontsize = 16)
        else:
            ax[1].set_ylabel('Coefficient Difference', fontsize = 25)
            ax[2].set_ylabel('Assignment Mismatch (%)', fontsize = 25)
            def format_func(value, tick_number):
                return f'{value:.1f}'
            ax[2].yaxis.set_major_formatter(ticker.FuncFormatter(format_func)) 
            ax[2].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        
        for i in ax:
            i.set_xlabel(Visualization.xlabel_mapper[parameter], fontsize = 25)
            if x_axis == 'time':
                ax[0].set_xlabel('Time (s)', fontsize = 25)
                ax[0].set_xscale('log')
            else:
                if parameter == 'outlier_ratio' or 'noise_std' in parameter:
                    i.set_xticks([float(i) for i in np.sort(common_parameters)][::2])
                else:
                    i.set_xticks([float(i) for i in np.sort(common_parameters)])

            i.grid(alpha = 0.3)
            i.tick_params(axis='x', labelsize=25)
            i.tick_params(axis='y', labelsize=25)
        
        if legend:
            ax[legend_position].legend(fontsize = 18)
        return fig, ax
    

    @staticmethod
    def plot_A_Ak_difference(
                parameter: str,
                mse_df_whiten: pd.DataFrame,
                mae_df_whiten: pd.DataFrame,
                optimal_df_whiten: pd.DataFrame,
                mse_df: pd.DataFrame,
                mae_df: pd.DataFrame,
                optimal_df: pd.DataFrame,
                mse_greedy = None,
                mae_greedy = None,
                x_axis = 'time',
                subset = [],
                methods = [],
                list_of_parameters = None):
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))
        ax =  [ax]
        data_dict = {'mse_whiten': mse_df_whiten, 'mae_whiten': mae_df_whiten, 'lpc_ns_mip': mse_df, 
                     'optimal_whiten': optimal_df_whiten, 'optimal_original': optimal_df, 'lpc_ns_qpbo': mae_df}
        
        if mse_greedy is not None:
            data_dict['mse_greedy'] = mse_greedy
            data_dict['mae_greedy'] = mae_greedy

        common_parameters = list_of_parameters
        for key, df in data_dict.items():
            data_dict[key][parameter] = data_dict[key][parameter].round(2)
            common_parameters = list(set(common_parameters).intersection(set(data_dict[key][parameter].unique())))
        for key, df in data_dict.items():
            data_dict[key] = data_dict[key][data_dict[key][parameter].isin(common_parameters)]   

        sem_dict = {}
        #rolling mean of each column
        for key, df in data_dict.items():
            result_df = df.copy()
            result_df_sem = df.copy()
            for i in range(result_df.shape[0]):
                for j in range(result_df.shape[1]):
                    if "±" in str(result_df.iloc[i, j]):
                        result_df.iloc[i, j] = float(result_df.iloc[i, j].split("±")[0])
                        result_df_sem.iloc[i, j] = float(result_df_sem.iloc[i, j].split("±")[1])
            result_df = result_df.astype(float)
            result_df_sem = result_df_sem.astype(float)
            sem_dict[key] = result_df_sem
            data_dict[key] = result_df

        for key, df in data_dict.items():
            result_df = df
            result_df_sem = sem_dict[key]
            if key == 'mse_whiten' or key == 'mae_whiten':
                models = ['greedy', 'milp', 'refit_milp_assignment', 'ground_truth']
            if key == 'lpc_ns_mip' and 'mse_greedy' not in data_dict.keys():
                models = ['greedy', 'milp', 'refit_milp_assignment', 'refit_ground_truth_assignment']
            elif key == 'lpc_ns_mip':
                models = ['milp', 'refit_milp_assignment', 'refit_ground_truth_assignment']
            if key == 'mse_greedy' or key == 'mae_greedy':
                models = ['greedy']
            if key == 'lpc_ns_qpbo' and 'mae_greedy' not in data_dict.keys():
                models = ['greedy', 'milp', 'refit_milp_assignment', 'refit_ground_truth_assignment']
            elif key == 'lpc_ns_qpbo':
                models = ['milp', 'refit_milp_assignment', 'refit_ground_truth_assignment']

            if key == 'optimal_whiten'or key == 'optimal_original':
                models = ['refit_milp_assignment']

            for i, column in enumerate(models):
                # mse
                tag = key.replace('_original', '').replace('_greedy', '') + '_' + column
                if subset != [] and tag in subset:
                    if x_axis == 'time':
                        if column == 'refit_ground_truth_assignment':
                            x = [0.1] * result_df.shape[0]
                            x_err = [0] * result_df.shape[0]
                        else:
                            x = result_df[f'time_{column}']
                            x_err = result_df_sem[f'time_{column}']
                    elif x_axis == 'v':
                            x = data_dict['optimal_original']['V_difference']
                            x_err = sem_dict['optimal_original']['V_difference']
                            
                    else:
                        x = result_df[parameter] 
                    for metric, plot_index in zip(['mse'], 
                                                 range(1)):
                        if metric == 'mse':
                            y = result_df[f'optimal_mse_difference_milp']
                            y_err = [0] * len(list_of_parameters)
                        else:
                            column_name =  f'{metric}_{column}'                           
                            y = result_df[column_name]
                            y_err = result_df_sem[column_name]
                        ax[plot_index].plot(x, y,
                                ls = '--', alpha = 0.8, 
                                c = Visualization.color_mapper[tag])
                        ax[plot_index].scatter(x, y, label = Visualization.label_mapper[tag], 
                                        s = 150, alpha = 0.4, c = Visualization.color_mapper[tag], 
                                        marker = Visualization.marker_mapper[tag])

                        if metric == 'mse' and tag == 'mse_refit_milp_assignment':
                            y = result_df['optimal_mse_difference_CLR']
                            y_err = [0] * len(list_of_parameters)
                        else:
                            column_name =  f'{metric}_{column}'                           
                            y = result_df[column_name]
                            y_err = result_df_sem[column_name]
                        ax[plot_index].plot(x, y,
                                ls = '--', alpha = 0.8, 
                                c = Visualization.color_mapper[tag])
                        ax[plot_index].scatter(x, y, label = 'CLR', 
                                        s = 150, alpha = 0.4, c = 'blue', 
                                        marker = 's')
                
       
        ax[0].set_ylabel('Difference From globalopt', fontsize = 20)
        a = '\\mathcal{E}_{sep}'
        ax[0].set_xlabel('$%s$'%a, fontsize = 20)
        #ax[0].set_yscale('log', base = 10)

        for i in ax:
            if x_axis == 'time':
                i.set_xlabel('Time (s)')
            i.grid(alpha = 0.3)
        ax[0].legend(loc = 'lower right', fontsize = 12, ncol = 1)

        return fig, ax
    
    @staticmethod
    def plot_trade_off(
            fig, ax,
            parameter: str,
            lpc_ns_mip: pd.DataFrame,
            lpc_ns_qpbo: pd.DataFrame,
            globalopt: pd.DataFrame,
            x_axis = 'time',
            subset = [],
            legend = True,
            list_of_parameters = None):

        data_dict = {'lpc_ns_mip': lpc_ns_mip, 'globalopt': globalopt, 'lpc_ns_qpbo': lpc_ns_qpbo}

        if list_of_parameters is not None:
            if parameter == 'D':
                list_of_parameters = np.array(list_of_parameters).astype(int)
            else:
                list_of_parameters = np.array(list_of_parameters).round(2)
            common_parameters = list_of_parameters
        else:
            common_parameters = list(set(lpc_ns_mip[parameter].round(2).unique()).intersection(set(lpc_ns_qpbo[parameter].round(2).unique())))
        
        for key, df in data_dict.items():
            if parameter == 'D':
                data_dict[key][parameter] = data_dict[key][parameter].astype(int)
            else:
                data_dict[key][parameter] = data_dict[key][parameter].round(2)
            common_parameters = list(set(common_parameters).intersection(set(data_dict[key][parameter].unique())))
        for key, df in data_dict.items():
            data_dict[key] = data_dict[key][data_dict[key][parameter].isin(common_parameters)].reset_index(drop = True)

        sem_dict = {}
        for key, df in data_dict.items():
            result_df = df.copy()
            result_df_sem = df.copy()
            for i in range(result_df.shape[0]):
                for j in range(result_df.shape[1]):
                    if "±" in str(result_df.iloc[i, j]):
                        result_df.iloc[i, j] = float(result_df.iloc[i, j].split("±")[0])
                        result_df_sem.iloc[i, j] = float(result_df_sem.iloc[i, j].split("±")[1])
            result_df = result_df.astype(float)
            result_df_sem = result_df_sem.astype(float)
            sem_dict[key] = result_df_sem
            data_dict[key] = result_df

        for key, df in data_dict.items():
            result_df = df
            result_df_sem = sem_dict[key]
            if key == 'lpc_ns_mip':
                models = ['greedy', 'milp', 'refit_milp_assignment']
            if key == 'lpc_ns_qpbo':
                models = ['greedy', 'milp', 'refit_milp_assignment']
            if key == 'globalopt':
                models = ['milp']

            for i, column in enumerate(models):
                tag = key + '_' + column
                if subset != [] and tag in subset:
                    if x_axis == 'time':
                        x = result_df[f'time_{column}']
                        x_err = result_df_sem[f'time_{column}']
                    else:
                        x = result_df[parameter] 
                        x_err = [0] * result_df.shape[0]

                    y = result_df['optimal_mse_difference_'+column]
                    y_err = result_df_sem['optimal_mse_difference_'+column]                          
                    ax.plot(x, y, label = Visualization.label_mapper[tag],
                                    linestyle='-', color='none', marker=Visualization.marker_mapper[tag],
                                    markerfacecolor='none',
                                    markeredgecolor=Visualization.color_mapper[tag], 
                                    markersize=20, alpha=0.8)
                    
                    for i in range(len(x) - 1):
                        ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]), 
                                    arrowprops=dict(arrowstyle='->,head_length=0.5,head_width=0.7', 
                                                    color=Visualization.color_mapper[tag], alpha=0.5,
                                                    linewidth=2, linestyle='--'))
                    for j in range(result_df.shape[0]):
                        ax.errorbar(x[j], y[j], xerr = x_err[j], yerr = y_err[j], marker = Visualization.marker_mapper[tag],
                            markerfacecolor='none', markersize=20, alpha = 0.8, c = Visualization.color_mapper[tag], capsize=3)
        
        # label and legend
        ax.set_ylabel('Diff From globalopt Obj', fontsize = 25)
        ax.set_xlabel(Visualization.xlabel_mapper[parameter], fontsize = 25)
        if x_axis == 'time':
            ax.set_xlabel('Time (s)', fontsize = 25)
            ax.set_xscale('log')

        ax.grid(alpha = 0.3)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=23)
        if legend:
            ax.legend(fontsize = 16, ncol = 1)

        return fig, ax
    
    @staticmethod
    def plot_time_vs_n(parameter, lpc_ns_mip: pd.DataFrame, lpc_ns_qpbo: pd.DataFrame, globalopt: pd.DataFrame, 
                    fig = None, ax = None, legend = False, list_of_plot_parameters = None):
            '''
            Plot the time taken by the models vs the parameter values

            Parameters:
            lpc_ns_mip: pd.DataFrame
                The dataframe containing the evaluation results in lpc_ns_mip
            lpc_ns_qpbo: pd.DataFrame
                The dataframe containing the evaluation results in lpc_ns_qpbo
            globalopt: pd.DataFrame
                The dataframe containing the optimal results in globalopt
            '''
            if fig is None and ax is None:
                fig, ax = plt.subplots(1, 1, figsize = (6, 5))
            data_dict = {'lpc_ns_mip': lpc_ns_mip, 'lpc_ns_qpbo': lpc_ns_qpbo, 'globalopt': globalopt}
            sem_dict = {}
            for key, _ in data_dict.items():
                sem_dict[key] = data_dict[key].copy()
                data = data_dict[key]
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if '±' in str(data.iloc[i, j]):
                            sem_dict[key].iloc[i, j] = float(data.iloc[i, j].split('±')[1])
                            data.iloc[i, j] = float(data.iloc[i, j].split('±')[0])
                            if data.iloc[i, j] > 7200:
                                data.iloc[i, j] = 7200
                                sem_dict[key].iloc[i, j] = 0

                for param in list_of_plot_parameters:
                    if param not in data[parameter].unique():
                        df = pd.DataFrame({parameter: [param], 'time_milp': [np.nan]})
                        sem_df = pd.DataFrame({parameter: [param], 'time_milp': [0]})
                        data = pd.concat([data, df], axis = 0)
                        data.reset_index(drop = True, inplace = True)
                        sem_dict[key] = pd.concat([sem_dict[key], sem_df], axis = 0)
                        sem_dict[key].reset_index(drop = True, inplace = True)

                data_dict[key] = data.astype(float)
                sem_dict[key] = sem_dict[key].astype(float)

            lpc_ns_mip = data_dict['lpc_ns_mip']
            lpc_ns_qpbo = data_dict['lpc_ns_qpbo']
            globalopt = data_dict['globalopt']

            lpc_ns_mip_sem = sem_dict['lpc_ns_mip']
            lpc_ns_qpbo_sem = sem_dict['lpc_ns_qpbo']
            globalopt_sem = sem_dict['globalopt']

            common_parameters = list(set(lpc_ns_qpbo[parameter].unique()).intersection(set(lpc_ns_mip[parameter].unique())))
            common_parameters = list(set(common_parameters).intersection(set(globalopt[parameter].unique())))
            if list_of_plot_parameters is not None:
                common_parameters = list(set(common_parameters).intersection(set(list_of_plot_parameters)))

            lpc_ns_mip = lpc_ns_mip[lpc_ns_mip[parameter].isin(common_parameters)].sort_values(by = parameter)
            lpc_ns_qpbo = lpc_ns_qpbo[lpc_ns_qpbo[parameter].isin(common_parameters)].sort_values(by = parameter)
            globalopt = globalopt[globalopt[parameter].isin(common_parameters)].sort_values(by = parameter)
            lpc_ns_mip_sem = lpc_ns_mip_sem[lpc_ns_mip_sem[parameter].isin(common_parameters)].sort_values(by = parameter)
            lpc_ns_qpbo_sem = lpc_ns_qpbo_sem[lpc_ns_qpbo_sem[parameter].isin(common_parameters)].sort_values(by = parameter)
            globalopt_sem = globalopt_sem[globalopt_sem[parameter].isin(common_parameters)].sort_values(by = parameter)

            x = lpc_ns_mip[parameter].values
            
            for i, column in enumerate(['refit_milp_assignment']):
                if 'time_'+column in lpc_ns_mip.columns:
                    ax.plot(x, lpc_ns_mip['time_'+column], 
                            ls = '--', alpha = 0.8, label = Visualization.label_mapper['lpc_ns_mip_'+column],
                              marker = Visualization.marker_mapper['lpc_ns_mip_'+column], 
                              markerfacecolor='none', markersize=20, 
                              c = Visualization.color_mapper['lpc_ns_mip_'+column])
                   
                    for j in range(lpc_ns_mip.shape[0]):
                        ax.errorbar(x[j], lpc_ns_mip['time_'+column].values[j], xerr = 0, yerr = lpc_ns_mip_sem['time_'+column].values[j],
                                    marker = Visualization.marker_mapper['lpc_ns_mip_'+column], 
                                    markerfacecolor='none', markersize=20, alpha = 0.8, 
                                    c = Visualization.color_mapper['lpc_ns_mip_'+column], capsize=5)
                        
                    ax.plot(x, lpc_ns_qpbo['time_milp'], label = Visualization.label_mapper['lpc_ns_qpbo_'+column],
                            ls = '--', alpha = 0.8, marker = Visualization.marker_mapper['lpc_ns_qpbo_'+column],
                            markerfacecolor='none', markersize=20, 
                            c = Visualization.color_mapper['lpc_ns_qpbo_'+column])
                    for j in range(lpc_ns_qpbo.shape[0]):
                        ax.errorbar(x[j], lpc_ns_qpbo['time_milp'].values[j], xerr = 0, yerr = lpc_ns_qpbo_sem['time_milp'].values[j], 
                                    marker = Visualization.marker_mapper['lpc_ns_qpbo_'+column], 
                                    markerfacecolor='none', markersize=20, alpha = 0.8, 
                                    c = Visualization.color_mapper['lpc_ns_qpbo_'+column], capsize=5)
                    
            ax.plot(x, lpc_ns_qpbo['time_greedy'], label = Visualization.label_mapper['lpc_ns_qpbo_greedy'],
                    ls = '--', alpha = 0.8, marker = Visualization.marker_mapper['lpc_ns_qpbo_greedy'], 
                    markerfacecolor='none', markersize=20, 
                    c = Visualization.color_mapper['lpc_ns_qpbo_greedy'])
            for i in range(lpc_ns_qpbo.shape[0]):
                ax.errorbar(x[i], lpc_ns_qpbo['time_greedy'].values[i], xerr = 0, yerr = lpc_ns_qpbo_sem['time_greedy'].values[i], 
                            marker = Visualization.marker_mapper['lpc_ns_qpbo_greedy'], 
                            markerfacecolor='none', markersize=20, alpha = 0.8, 
                            c = Visualization.color_mapper['lpc_ns_qpbo_greedy'], capsize=5)
            
            ax.plot(x, globalopt['time_milp'], label = Visualization.label_mapper['globalopt_milp'],
                    ls = '--', alpha = 0.8, marker = Visualization.marker_mapper['globalopt_milp'],
                    markerfacecolor='none', markersize=20, c = Visualization.color_mapper['globalopt_milp'])
            for i in range(globalopt.shape[0]):
                ax.errorbar(x[i], globalopt['time_milp'].values[i], xerr = 0, yerr = globalopt_sem['time_milp'].values[i], 
                            marker = Visualization.marker_mapper['globalopt_milp'], 
                            markerfacecolor='none', markersize=20, alpha = 0.8, 
                            c = Visualization.color_mapper['globalopt_milp'], capsize=5)

            # label and legend
            ax.set_ylabel('Time (s)', fontsize = 25)
            ax.set_yscale('log')
            ax.set_xlabel(Visualization.xlabel_mapper[parameter], fontsize = 25)
            #set xtick font size to 14
            ax.set_ylim(0.1, 10000)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=25)
            ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            #rotate x-axis labels
            plt.xticks(rotation=45)
            if legend:
                plt.legend(fontsize = 20, ncol = 1)
            #grid
            ax.grid(True, alpha = 0.3)  
            return fig, ax