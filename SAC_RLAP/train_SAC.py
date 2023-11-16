import os
import json
import random
import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from SAC_RLAP.helpers import create_env
from SAC_RLAP.SACs import SAC
from SAC_RLAP.evaluate import evaluate


def train(args, load_name=None):
    writer = SummaryWriter(args.run_dir)

    env = create_env(args, args.training_patients[0])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    agent = SAC(args, env.observation_space.shape[0], env.action_space.shape[0])
    if (load_name is not None) and (load_name != '') and (load_name != ' '): agent.load(f'./models/{load_name}/')
    
    training_patients = cycle(args.training_patients)
    patient_name = next(training_patients)
    best_score, patient_time_stamp, video_time_stamp, evaluate_time_stamp = 0, 0, 0, 0

    # For SAC_PER_ERE
    eta_0 = 0.996
    eta_T = 1.0
    c_k_min = 2500

    while args.current_timestep < args.max_total_timesteps:

        if len(args.training_patients) != 1:  # Ensemble training
            if args.current_timestep - patient_time_stamp >= args.max_episode_timesteps:  # Switch patient
                patient_name = next(training_patients)
                env = create_env(args, patient_name)
                env.seed(args.seed + args.current_episode)  # To prevent the same environments reoccuring
                patient_time_stamp = args.current_timestep

        state, done, ep_reward = env.reset(), False, 0
        args.current_episode += 1

        for t in range(1, args.max_episode_timesteps+1):
            
            args.current_timestep += 1
            
            if args.current_timestep - video_time_stamp > args.video_every: env.render()

            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

            eta_t = eta_0 + (eta_T - eta_0)*(t/args.current_timestep)  # For SAC_PER_ERE

            if (args.model_name != 'SAC_PER_ERE') and (len(agent.memory) > args.batch_size): 
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters()
            
            elif (args.model_name == 'SAC_PER_ERE') and (done or t==args.max_episode_timesteps):
                for k in range(1, t):
                    c_k = max(int(agent.memory.__len__()*eta_t**(k*(args.max_episode_timesteps/t))), c_k_min)
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(c_k)
            
            else: critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = 0, 0, 0, 0, args.alpha

            if done or t==args.max_episode_timesteps:
                writer.add_scalar('loss/critic_1', critic_1_loss, args.current_timestep)
                writer.add_scalar('loss/critic_2', critic_2_loss, args.current_timestep)
                writer.add_scalar('loss/policy_loss', policy_loss, args.current_timestep)
                writer.add_scalar('loss/entropy_loss', ent_loss, args.current_timestep)
                writer.add_scalar('training/entropy_temperature', alpha, args.current_timestep)
                writer.add_scalar('training/timesteps', t, args.current_timestep)
                writer.add_scalar('training/average_reward', ep_reward/t, args.current_timestep)
                writer.add_scalar('training/cumulative_reward', ep_reward, args.current_timestep)
                print(f'{patient_name} | episode {args.current_episode} | timesteps {t} | avg_reward {ep_reward/t} | t_total {args.current_timestep}')

                if args.current_timestep - video_time_stamp > args.video_every: plt.show(); video_time_stamp = args.current_timestep

                if args.current_timestep - evaluate_time_stamp > args.evaluate_every: 
                    evaluate_time_stamp = args.current_timestep

                    if args.save_mode == 'every':
                        # Always satisfies condition below since best_score is 0
                        new_score = 1
                    if args.save_mode == 'best':
                        # Average episode reward (with respect to max episode length not actual episode length)
                        new_score = ep_reward / args.max_episode_timesteps
                    if args.save_mode == 'validate':
                        val_score = evaluate(args, agent, args.validation_patients)
                        writer.add_scalar('training/validation_score', val_score, args.current_timestep)
                        # Reciprocal of risk ensures that lower risks are give a higher score and are always greater than 0
                        new_score = 1 / val_score

                    if new_score >= best_score:
                        # Update score
                        best_score = new_score

                        # Save hyperparameters
                        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
                        with open(args.save_dir + 'hyperparameters.txt', 'w') as f:
                            json.dump(args.__dict__, f, indent=2)

                        # Save model
                        agent.save(args.save_dir) 

                break