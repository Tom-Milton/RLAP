import pandas as pd
from tqdm import tqdm
from IPython import display
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from simglucose.analysis.report import report
from simglucose.simulation.sim_engine import SimObj, batch_sim

from SAC_RLAP.helpers import create_env
from SAC_RLAP.SACs import SAC


def evaluate(args, agent, patient_names, num_simulations=1, num_timesteps=1440):
    # Average total reward of num_simulations (1) of length num_timesteps (3 days) for each patient
    total_risk = 0
    
    for patient_name in patient_names:
        env = create_env(args, patient_name)
        env.seed(args.seed)

        for _ in range(num_simulations):
            state = env.reset()

            for _ in range(1, num_timesteps+1):
                action = agent.select_action(state, evaluate=True)
                state, _, _, _ = env.step(action)
            
            df = env.show_history()
            total_risk += df['Risk'].sum()

    return total_risk/(len(patient_names)*num_simulations*num_timesteps)


def play(args, agent, patient_name):
    args.start_time = datetime(2018,1,1,0,0,0)
    env = create_env(args, patient_name)

    state = env.reset()
    for _ in range(1, args.max_episode_timesteps+1):
        env.render()
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = agent.select_action(state, evaluate=True)
        state, _, _, _ = env.step(action)
    plt.show()


def results(args, agent, patient_names):
    args.start_time = datetime(2018,1,1,0,0,0)

    s = []
    for patient in patient_names:
        env = create_env(args, patient)
        env.seed(args.seed)
        s.append(SimObj(env, agent, timedelta(days=10), animate=False, path=args.results_dir))
    
    batch_sim(s)


def print_results(args, patient_names):
    df = pd.concat([pd.read_csv(args.results_dir + patient + '.csv', index_col=0) for patient in patient_names], keys=patient_names)
    df.sort_index(inplace=True)
    results, _, _, _, _ = report(df)
    train_results = results.loc[args.training_patients]
    val_results = results.loc[args.validation_patients]
    test_results = results.loc[args.testing_patients]

    print('Mean Avg TIR: ' + str(results['70<=BG<=180'].mean()))
    print('Mean Avg Hypo: ' + str(results['BG<70'].mean()))
    print('Mean Avg Hyper: ' + str(results['BG>180'].mean()))
    print('Mean Avg Risk: ' + str(results['Risk Index'].mean()))
    print('')

    print('Mean Train TIR: ' + str(train_results['70<=BG<=180'].mean()))
    print('Mean Train Hypo: ' + str(train_results['BG<70'].mean()))
    print('Mean Train Hyper: ' + str(train_results['BG>180'].mean()))
    print('Mean Train Risk: ' + str(train_results['Risk Index'].mean())) 
    print('')

    print('Mean Val TIR: ' + str(val_results['70<=BG<=180'].mean()))
    print('Mean Val Hypo: ' + str(val_results['BG<70'].mean()))
    print('Mean Val Hyper: ' + str(val_results['BG>180'].mean()))
    print('Mean Val Risk: ' + str(val_results['Risk Index'].mean()))   
    print('')

    print('Mean Test TIR: ' + str(test_results['70<=BG<=180'].mean()))
    print('Mean Test Hypo: ' + str(test_results['BG<70'].mean()))
    print('Mean Test Hyper: ' + str(test_results['BG>180'].mean()))
    print('Mean Test Risk: ' + str(test_results['Risk Index'].mean()))