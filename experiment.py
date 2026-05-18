"""
Experiment file used to get the averaged results of multiple runs of REINFORCE, AC and A2C, creating csv files with the averages
and plotting themm together.
"""
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import csv

from scipy.signal import savgol_filter
from agents import *

def get_full_run_results(file_name,learner_type:ModelFreeLearner,num_repetitions = 5, actor_lr = 1e-4, critic_lr = 1e-4, budget =  1000000, 
                         epsilon=0.2, entropy_coefficient = 0.01, epochs = 5,minibatch_len=100, rollout_len=2000, target_kl=0.02):  
    """
    Executes the optimization process for the correspondent type of ModelFreeLearner, and generates a csv file with the averaged resuls.
    """  
    # specify the directory name
    #directory_name = 'Hyper_parameter_testing'
    directory_name = 'Full_Run_Results'

    # create the directory if it dosnt exist
    if not os.path.isdir(directory_name):
        os.mkdir(directory_name)

    file_path = f'{directory_name}/{file_name}.csv'

    if os.path.exists(file_path):
        # prevents rerunning experiments
        print(f'Experiment already completed! Results at: {file_path}')
        return
    
    print(f'Running experiment: {file_path}')

    results = []
    
    #loop over number of repetitions for averaging
    for i in range(num_repetitions):
        curr_env = gym.make("CartPole-v1")
        
        #Initialize learner depending on the experiment
        if file_name == "PPO":
            learner = learner_type(curr_env,2,2,0.99, actor_lr,critic_lr, True, epsilon=epsilon, 
                                   entropy_coefficient = entropy_coefficient, epochs = epochs, minibatch_len = minibatch_len, target_kl = target_kl)
        else:
            learner = learner_type(curr_env,2,2,0.99, actor_lr,critic_lr)

        #Get optimization results
        evaluation = learner.optimize(budget)
        if len(evaluation) > (budget/250):
            evaluation = evaluation[:-int(len(evaluation) - (budget/250))]
        curr_eval_returns, curr_eval_timesteps  = [*zip(*evaluation)]
        print(len(np.array(curr_eval_returns)))
        results.append(np.array(curr_eval_returns))
    
    results = np.array(results)
    #export evaluation to a csv file
    df = pd.DataFrame({
        "eval_timesteps": list(map(int, curr_eval_timesteps)),
        "eval_mean_returns":np.mean(results, axis=0),
        "eval_std_returns":np.std(results, axis=0)
        })
    df.to_csv(file_path)

def plot_full_runs(solved_threshold=500, num_repetitions = 5):
    """
    Goes through the Full_Run_Results folder and plots the results of each csv file in it
    """
    items = []
    #folder = f'Hyper_parameter_testing/'
    folder = f'Full_Run_Results/'
    files = os.listdir(folder)
    index = 0

    #iterate through the files getting the file name for the plot title, timesteps, results and standard deviation, saving it as an item
    while index < len(files):
        x = []
        y = []
        std = []
        filename = files[index]
        if filename.endswith('.csv'):
            with open(f'{folder}{filename}','r') as csvfile:
                lines = csv.reader(csvfile, delimiter=',')
                title = ""
                for row in lines:
                    if row[1] == 'eval_timesteps':
                        title =  filename.split(".")[0]
                    else:
                        x.append(int(row[1]))
                        y.append(float(row[2]))
                        std.append(float(row[3]))
            items.append({'label':title,'x':x,'y':y,'std':std})
        index +=1
    #define smoothing window
    smoothing_window = 81
    colors = ['#56B4E9','#9467bd','#ff7f0e', '#009E73', '#E69F00', '#F0E442', '#0072B2', '#D55E00', '#CC79A7','#1f77b4','#2ca02c','#d62728']
    fig, ax = plt.subplots(figsize=(8, 6))
    #plot for every item collected, adding smoothing with a savgol_filter
    for item, color in zip(items, colors):
        smooth = savgol_filter(item['y'],smoothing_window,2)
        err = item["std"]/np.sqrt(num_repetitions)
        err_smooth = savgol_filter(err,smoothing_window,2)
        ax.plot(item["x"], smooth, color=color, linewidth=2, label=item["label"])
        ax.fill_between(item["x"],smooth-err_smooth,smooth+err_smooth,alpha=0.2, color=color)
        
    #plot max reward threshold
    ax.axhline(solved_threshold, color="#E24B4A", linewidth=1.5,
               linestyle="--", label=f"max reward ({solved_threshold})")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Learning curves in the CartPole environment")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()

#select a seed to make results replicable
torch.manual_seed(2001)
#Hyperparameter testing
#epsilon_list=[0.05,0.1,0.2]
#entropy_coefficient_list = [0.1,0.01,0.001]
#epochs_list = 5
#minibatch_len_list=[50,100]
#target_kl_list=[0.01,0.02,0.05]
#for epsilon in epsilon_list:
#    for entropy_coefficient in entropy_coefficient_list:
#        for minibatch_len in minibatch_len_list:
#            for target_kl in target_kl_list:
#                get_full_run_results(f'PPO_epsilon{epsilon}_ec{entropy_coefficient}_mblen{minibatch_len}_kl{target_kl}', PPO,num_repetitions=2, budget = 200000
#                                     ,epsilon=epsilon,entropy_coefficient=entropy_coefficient,minibatch_len=minibatch_len,target_kl=target_kl)

#Full Runs
get_full_run_results(f'PPO', PPO,num_repetitions=5, budget = 1000000
                    ,epsilon=0.05,entropy_coefficient=0.01,minibatch_len=50,target_kl=0.01)
#get_full_run_results(f'ep_005_ec01_mblen100_kl005', PPO,num_repetitions=5, budget = 1000000
#                    ,epsilon=0.05,entropy_coefficient=0.1,minibatch_len=100,target_kl=0.05)
#get_full_run_results(f'ep_01_ec01_mblen100_kl001', PPO,num_repetitions=5, budget = 1000000
#                    ,epsilon=0.1,entropy_coefficient=0.1,minibatch_len=100,target_kl=0.01)
#get_full_run_results(f'ep_01_ec0001_mblen50_kl001', PPO,num_repetitions=5, budget = 1000000
#                    ,epsilon=0.1,entropy_coefficient=0.001,minibatch_len=50,target_kl=0.01)
#get_full_run_results(f'ep_02_ec001_mblen50_kl002', PPO,num_repetitions=5, budget = 1000000
#                    ,epsilon=0.2,entropy_coefficient=0.01,minibatch_len=50,target_kl=0.02)
#get_full_run_results(f'ep_02_ec0001_mblen50_kl005', PPO,num_repetitions=5, budget = 1000000
#                    ,epsilon=0.2,entropy_coefficient=0.001,minibatch_len=50,target_kl=0.05)
plot_full_runs()



