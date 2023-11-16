import json
import argparse
from datetime import datetime


def hyperparameters(load_name=None):
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--current_episode', default=0, type=int, help="Current episode during the training process")
    parser.add_argument('--current_timestep', default=0, type=int, help="Current timestep during the training process")
    parser.add_argument('--max_episode_timesteps', default=2_400, type=int, help="Maximum number of timesteps per episode - currently set to 5 days")
    parser.add_argument('--max_total_timesteps', default=100_000, type=int, help="Total number of timesteps for training")
    parser.add_argument('--video_every', default=100_000, type=int, help="How many timesteps before producing an animation")
    parser.add_argument('--evaluate_every', default=10_000, type=int, help="How many timesteps before checking to save the model")
    parser.add_argument('--save_mode', default="every", type=str, help="Method to check if saving the model after evaluate_every timesteps. 'every' saves every time, 'best' saves if a new best reward is seen, 'validate' saves if new lowest risk on validation cohort")
    parser.add_argument('--seed', default=42, type=int, help="Random seet for reproducibility")
    parser.add_argument('--save_dir', default="./models/SAC_model/", type=str, help="Directory to save the model to")
    parser.add_argument('--results_dir', default="./results/SAC_results/", type=str, help="Directory to save the model's results to")
    parser.add_argument('--run_dir', default="./runs/SAC_run/", type=str, help="Directory to save the model's tensorboard logs to")

    # Environment Parameters
    parser.add_argument('--training_patients', default=['adult#001'], type=list, help="List of training patients")
    parser.add_argument('--validation_patients', default=['adult#001'], type=list, help="List of validation patients")
    parser.add_argument('--testing_patients', default=['adult#001'], type=list, help="List of testing patients")
    parser.add_argument('--custom_scenario', default='harris_benedict', type=str, help="Custom meal scenario. 'no_meals' has no meals. 'random_scaled' scales meal sizes with respect to the patient's body weight. 'harris_benedict' scales meal sizes with respect to the Harris Benedict equation that considers the patient's weight, age, and height")
    parser.add_argument('--risk_metric', default='clarke', type=str, help="Either clarke or magni risk metric")
    parser.add_argument('--start_time', default=None, type=datetime, help="Specify fixed start time. 'None' chooses random start times")
    parser.add_argument('--normalise_state_space', default=True, type=bool, help="Normalise state space to between 0 and 1")
    parser.add_argument('--normalise_action_space', default=True, type=bool, help="Normalise action space to between 0 and 1")
    parser.add_argument('--meal_state', default=False, type=bool, help="Add ground truth meals to state space")
    parser.add_argument('--weight_state', default=False, type=bool, help="Add patient's body weight to state space")
    parser.add_argument('--height_state', default=False, type=bool, help="Add patient's height to state space")
    parser.add_argument('--age_state', default=False, type=bool, help="Add patient's age to state space")
    parser.add_argument('--tdi_state', default=False, type=bool, help="Add patient's total daily insulin to state space")
    parser.add_argument('--pid_state', default=False, type=bool, help="Selects PID state space given by the last blood glucose reading. Replaces all other state space choices")
    parser.add_argument('--n_hours', default=4, type=float, help="Number of hours of past blood glucose, insulin, and meals data for the state space elements")
    parser.add_argument('--insulin_max', default=0.3, type=float, help="Action range upper bound given by the possible maximum insulon dosage")
    parser.add_argument('--scale_factor', default=0.1, type=float, help="Action range lower bound given by negative scale_factor percent of insulin_max - currently set to -0.03")

    # Agent Parameters
    parser.add_argument("--model_name", default="SAC", type=str, help="Model name either SAC, SAC_PER, SAC_PER_ERE, or PID")
    parser.add_argument("--device", default="cpu", type=str, help="Run torch on cpu or gpu")
    parser.add_argument('--alpha', default=0.2, type=float, help="Entropy temperature (exploration)")
    parser.add_argument('--tau',  default=0.005, type=float, help="Soft polyak update coefficient")
    parser.add_argument('--lr', default=3e-4, type=float, help="Learning rate for actor and critic")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor for future rewards")
    parser.add_argument('--buffer_size', default=1_000_000, type=int, help="Size of the replay buffer")
    parser.add_argument('--batch_size', default=256, type=int, help="Size of the batches during training")
    parser.add_argument('--hidden_size', default=256, type=int, help="Number of neurons in the hidden layer for the actor and critic")
    parser.add_argument('--automatic_entropy_tuning', default=True, type=bool, help="Automatically adjust exploration (alpha (temperature)) during training")
    parser.add_argument('--lr_schedule', default=True, type=bool, help="Increase learning rate for actor and critic linearly for the first 10_000 timesteps at the start of training")

    args, unknown = parser.parse_known_args()

    if (load_name is not None) and (load_name != '') and (load_name != ' '):
        with open(f'./models/{load_name}/' + 'hyperparameters.txt', 'r') as f: 
            args.__dict__ = json.load(f)

    return args