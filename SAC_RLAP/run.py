import warnings
import pandas as pd

from SAC_RLAP.hyperparameters import hyperparameters
from SAC_RLAP.train_SAC import train
from SAC_RLAP.tune_PID import tune
from SAC_RLAP.evaluate import play, results, print_results
from SAC_RLAP.helpers import create_env
from SAC_RLAP.SACs import SAC

from simglucose.controller.pid_ctrller import PIDController


def run(model, scenario, save_mode, timesteps, patient_classes, patient_selection, state_space, processes, save_name, load_name):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning) 

    args = hyperparameters(load_name)
    args.model_name = model
    args.custom_scenario = scenario
    args.save_mode = save_mode
    args.max_total_timesteps = timesteps

    patient_classes = [x for x, y in zip(['adult#', 'adolescent#', 'child#'], patient_classes) if y]
    args.training_patients = [type + str(number).zfill(3) for number in range(patient_selection[0], patient_selection[1]) for type in patient_classes]
    args.validation_patients = [type + str(number).zfill(3) for number in range(patient_selection[1], patient_selection[2]) for type in patient_classes]
    args.testing_patients = [type + str(number).zfill(3) for number in range(patient_selection[2], patient_selection[3]) for type in patient_classes]
    all_patients = args.training_patients + args.validation_patients + args.testing_patients

    print('training patients:', args.training_patients)
    print('validation patients:', args.validation_patients)
    print('testing patients:', args.testing_patients)

    # if 'child#' in patient_classes: args.insulin_max = 0.1
    # if 'adolescent#' in patient_classes: args.insulin_max = 0.2
    # if 'adult#' in patient_classes: args.insulin_max = 0.3

    if args.model_name == 'PID':
        args.normalise_state_space = False
        args.normalise_action_space = False
        args.pid_state = True
        args.results_dir = './results/PID_model/'
    
        if processes[0]:
            tune(args)

        if processes[1]: 
            pid_parameters = pd.read_csv('./SAC_RLAP/pid_parameters.csv', index_col='Name')
            p, i, d = pid_parameters.loc[all_patients[0]].tolist()
            agent = PIDController(P=p, I=i, D=d, target=112)
            play(args, agent, all_patients[0])

        if processes[2]: 
            for patient_name in all_patients:
                pid_parameters = pd.read_csv('./SAC_RLAP/pid_parameters.csv', index_col='Name')
                p, i, d = pid_parameters.loc[patient_name].tolist()
                agent = PIDController(P=p, I=i, D=d, target=112)
                results(args, agent, [patient_name])

        if processes[3]: 
            print_results(args, all_patients)

    else:
        if processes[0]:
            # Setting hyperparameters
            args.weight_state = state_space[0]
            args.age_state = state_space[1]
            args.height_state = state_space[2]
            args.tdi_state = state_space[3]
            args.meal_state = state_space[4]

            if (load_name is not None) and (load_name != '') and (load_name != ' '):
                args.current_episode = 0
                args.current_timestep = 0

            # Creating preset run name
            patients_classes = ['adult', 'adolescent', 'child']
            patients_numbers = ['_', '_', '_'] 
            for patient_name in args.training_patients:
                if patient_name[:-4] == 'adult': patients_numbers[0] = patients_numbers[0] + patient_name[-1]
                if patient_name[:-4] == 'adolescent': patients_numbers[1] = patients_numbers[1] + patient_name[-1]
                if patient_name[:-4] == 'child': patients_numbers[2] = patients_numbers[2] + patient_name[-1]
            patients_run_name = [a + b for a, b in zip(patients_classes, patients_numbers)]

            state_space_elements = ['weight', 'age', 'height', 'tdi', 'meals']
            state_space_run_name = [x for x, y in zip(state_space_elements, state_space) if y]

            # Check for custom run_name asign create preset run_name
            if (save_name is not None) and (save_name != '') and (save_name != ' '): run_name = save_name
            else: run_name = f'{args.model_name}_{args.custom_scenario}_{args.save_mode}_{args.max_total_timesteps}_{patients_run_name}_{state_space_run_name}'
            print('run_name:', run_name)
            
            args.save_dir = f'./models/{run_name}/'
            args.results_dir = f'./results/{run_name}/'
            args.run_dir = f'./runs/{run_name}/'

            train(args, load_name)
        
        # Load pre-trained agent (either from training or loading)
        env = create_env(args, all_patients[0])
        agent = SAC(args, env.observation_space.shape[0], env.action_space.shape[0])
        agent.load(args.save_dir)
        
        if processes[1]: play(args, agent, all_patients[0])

        if processes[2]: results(args, agent, all_patients)

        if processes[3]: print_results(args, all_patients)