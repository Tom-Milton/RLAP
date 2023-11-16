import gym
import numpy as np

from simglucose.envs.simglucose_gym_env import T1DSimEnv

from simglucose.analysis.risk import clarke_risk_index, magni_risk_index


def create_env(args, patient_name, reward_fun=None):
    if reward_fun is not None: reward_fun = reward_fun
    elif args.risk_metric == 'clarke': reward_fun = clarke_risk_reward
    elif args.risk_metric == 'magni': reward_fun = magni_risk_reward

    return T1DSimEnv(patient_name=patient_name, custom_scenario=args.custom_scenario, risk_metric=args.risk_metric, reward_fun=reward_fun, 
                    start_time=args.start_time, normalise_state_space=args.normalise_state_space, normalise_action_space=args.normalise_action_space,
                    weight_state=args.weight_state, height_state=args.height_state, age_state=args.age_state, tdi_state=args.tdi_state, meal_state=args.meal_state, 
                    pid_state=args.pid_state, n_hours=args.n_hours, insulin_max=args.insulin_max, scale_factor=args.scale_factor)
    

def normalise(x, new_min, new_max, old_min, old_max):
    a = (new_max - new_min)/(old_max - old_min)
    b = x - old_min
    c = new_min
    return a*b + c


def clarke_risk_reward(risk_hist, **kwargs):
    max_risk = max(clarke_risk_index(50)[2], clarke_risk_index(350)[2])
    reward = max_risk - np.clip(risk_hist[-1], 0, max_risk)
    reward = normalise(reward, 0, 1, 0, max_risk)
    return reward


def magni_risk_reward(risk_hist, **kwargs):
    max_risk = max(magni_risk_index(50)[2], magni_risk_index(350)[2])
    reward = max_risk - np.clip(risk_hist[-1], 0, max_risk)
    reward = normalise(reward, 0, 1, 0, max_risk)
    return reward