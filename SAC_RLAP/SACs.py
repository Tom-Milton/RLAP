import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import LinearLR

from SAC_RLAP.actor_critic import Actor, Critic
from SAC_RLAP.replay_buffers import ReplayBuffer, PrioritizedReplay, PrioritizedReplayERE


class SAC():    
    """https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py"""
    def __init__(self, args, state_size, action_size, action_prior="uniform"):
        self.args = args
        self.state_size = state_size
        self.action_size = action_size
        
        # Automatic entropy tuning
        self.alpha = args.alpha
        self.target_entropy = -action_size  # -dim(A)
        self.log_alpha = torch.tensor([math.log(self.alpha)], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=args.lr) 
        self._action_prior = action_prior
                
        # Actor Network 
        self.policy = Actor(state_size, action_size, args.device, args.hidden_size).to(args.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)     
        
        # Critic Network
        self.critic1 = Critic(state_size, action_size, args.device, args.hidden_size).to(args.device)
        self.critic2 = Critic(state_size, action_size, args.device, args.hidden_size).to(args.device)
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr) 

        # Target Network
        self.critic1_target = Critic(state_size, action_size, args.device, args.hidden_size).to(args.device)
        self.hard_update(self.critic1_target, self.critic1)

        self.critic2_target = Critic(state_size, action_size, args.device, args.hidden_size).to(args.device)
        self.hard_update(self.critic2_target, self.critic2) 

        # Learning Rate Scheduler
        if args.lr_schedule:
            self.policy_scheduler = LinearLR(self.policy_optimizer, start_factor=1e-6, total_iters=1e4)
            self.critic1_scheduler = LinearLR(self.critic1_optimizer, start_factor=1e-6, total_iters=1e4)
            self.critic2_scheduler = LinearLR(self.critic2_optimizer, start_factor=1e-6, total_iters=1e4)

        # Replay memory
        if args.model_name == 'SAC': self.memory = ReplayBuffer(args.buffer_size, args.device)
        elif args.model_name == 'SAC_PER': self.memory = PrioritizedReplay(args.buffer_size, args.device)
        elif args.model_name == 'SAC_PER_ERE': self.memory = PrioritizedReplayERE(args.buffer_size, args.device)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
             
    def select_action(self, state, evaluate=False):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.args.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, c_k=None):
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.args.batch_size, c_k)      

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next, _ = self.policy.sample(next_states)

        Q_target1_next = self.critic1_target(next_states.to(self.args.device), next_action.squeeze(0).to(self.args.device))
        Q_target2_next = self.critic2_target(next_states.to(self.args.device), next_action.squeeze(0).to(self.args.device))

        # Take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        
        # Compute Q targets for current states (y_i)
        if self.args.automatic_entropy_tuning:
            Q_targets = rewards.cpu() + (self.args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (self.args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.squeeze(0).cpu()))

        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        td_error1 = Q_targets.detach()-Q_1
        td_error2 = Q_targets.detach()-Q_2
        critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
        critic2_loss = 0.5* (td_error2.pow(2)*weights).mean()
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.memory.update_priorities(indices, prios.data.cpu().numpy())

        # ---------------------------- update actor and entropy ---------------------------- #
        if self.args.automatic_entropy_tuning:
            alpha = torch.exp(self.log_alpha)
            actions_pred, log_pis, _ = self.policy.sample(states)
            alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
            self.args.alpha = self.alpha.detach().item()  # For hyperparameters.txt logs

            # Compute actor loss
            if self._action_prior == "normal":
                policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                policy_prior_log_probs = policy_prior.log_prob(actions_pred)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0

            policy_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs).mean()

        else:
            actions_pred, log_pis, _ = self.policy.sample(states)
            if self._action_prior == "normal":
                policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                policy_prior_log_probs = policy_prior.log_prob(actions_pred)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0

            policy_loss = (self.args.alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs).mean()
            
        # Minimize the loss
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, self.args.tau)
        self.soft_update(self.critic2, self.critic2_target, self.args.tau)

        # Take scheduler steps
        if self.args.lr_schedule:
            self.policy_scheduler.step()
            self.critic1_scheduler.step()
            self.critic2_scheduler.step()

        return critic1_loss.item(), critic2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def save(self, save_dir):
        torch.save(self.policy.state_dict(), save_dir + 'policy.pth')
        torch.save(self.policy_optimizer.state_dict(), save_dir + 'policy_optimizer.pth')
        torch.save(self.critic1.state_dict(), save_dir + 'critic1.pth')
        torch.save(self.critic2.state_dict(), save_dir + 'critic2.pth')
        torch.save(self.critic1_target.state_dict(), save_dir + 'critic1_target.pth')
        torch.save(self.critic2_target.state_dict(), save_dir + 'critic2_target.pth')
        torch.save(self.critic1_optimizer.state_dict(), save_dir + 'critic1_optimizer.pth')
        torch.save(self.critic2_optimizer.state_dict(), save_dir + 'critic2_optimizer.pth')
        print("Model has been saved...")

    def load(self, load_dir):
        self.policy.load_state_dict(torch.load(load_dir + 'policy.pth'))
        self.policy_optimizer.load_state_dict(torch.load(load_dir + 'policy_optimizer.pth'))
        self.critic1.load_state_dict(torch.load(load_dir + 'critic1.pth'))
        self.critic2.load_state_dict(torch.load(load_dir + 'critic2.pth'))
        self.critic1_target.load_state_dict(torch.load(load_dir + 'critic1_target.pth'))
        self.critic2_target.load_state_dict(torch.load(load_dir + 'critic2_target.pth'))
        self.critic1_optimizer.load_state_dict(torch.load(load_dir + 'critic1_optimizer.pth'))
        self.critic2_optimizer.load_state_dict(torch.load(load_dir + 'critic2_optimizer.pth'))
        print("Model has been loaded...")

        # Need to reset lr_schedulers
        if self.args.lr_schedule:
            self.policy_scheduler = LinearLR(self.policy_optimizer, start_factor=1e-6, total_iters=1e4)
            self.critic1_scheduler = LinearLR(self.critic1_optimizer, start_factor=1e-6, total_iters=1e4)
            self.critic2_scheduler = LinearLR(self.critic2_optimizer, start_factor=1e-6, total_iters=1e4)