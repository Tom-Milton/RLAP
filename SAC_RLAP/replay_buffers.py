import torch
import random
import numpy as np
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.memory = deque(maxlen=capacity)  # internal memory (deque)
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def push(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size, c_k=None):
        indices = random.sample(self.memory, k=batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in indices if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in indices if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in indices if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in indices if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in indices if e is not None]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(np.vstack([1.0 for e in indices if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        pass
    
    def __len__(self):
        return len(self.memory)


class PrioritizedReplay(object):
    def __init__(self, capacity, device, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    
    def sample(self, batch_size, c_k=None):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        states      = torch.FloatTensor(np.float32(np.concatenate(states))).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device) 
        actions     = torch.FloatTensor(np.concatenate(actions)).to(self.device).unsqueeze(1) 
        rewards     = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones       = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights     = torch.FloatTensor(weights).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 

    def __len__(self):
        return len(self.buffer)

        
class PrioritizedReplayERE(object):
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.pos        = 0
        self.priorities = deque(maxlen=capacity)
    
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = max(self.priorities) if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        self.buffer.insert(0, (state, action, reward, next_state, done))
        self.priorities.insert(0, max_prio)
    
    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k > N:
            c_k = N
        if N == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(list(self.priorities)[:c_k])
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p and the c_k range of the buffer
        indices = np.random.choice(c_k, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        states      = torch.FloatTensor(np.float32(np.concatenate(states))).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device) 
        actions     = torch.FloatTensor(np.concatenate(actions)).to(self.device).unsqueeze(1) 
        rewards     = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones       = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights     = torch.FloatTensor(weights).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 

    def __len__(self):
        return len(self.buffer)