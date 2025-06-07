"""
This file implements PPO algorithm.

-----
Based of code from: 

CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.

Modified by: 
"""
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add_transition(self, state, action, logprob, state_value, reward, is_terminal):
        # This method will be called after each step, adding a complete transition
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(is_terminal)

    def add_pending_transition(self, state, action, logprob, state_value):
        # Adds a transition for which reward and is_terminal are not yet known
        self.states.append(state) # state is already a tensor e.g. (1, state_dim)
        self.actions.append(action) # action is already a tensor e.g. (1, action_dim)
        self.logprobs.append(logprob) # logprob is already a tensor e.g. (1,)
        self.rewards.append(None) # Placeholder for reward
        self.state_values.append(state_value) # state_value is already a tensor e.g. (1,)
        self.is_terminals.append(None) # Placeholder for is_terminal

    def update_last_transition(self, reward, is_terminal):
        # Updates the reward and is_terminal for the last added transition
        if not self.rewards or self.rewards[-1] is not None:
            print("Warning: Attempting to update a non-pending or non-existent transition.")
            # Optionally, handle this case by adding a new full transition if it's intended
            # For now, we assume this is called correctly after add_pending_transition
            return 
        self.rewards[-1] = reward
        self.is_terminals[-1] = is_terminal

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        
        # state_values are stored as tensors (e.g., shape (1,) or scalar tensors)
        # Stack them to create a single tensor of shape (num_steps,)
        if not self.state_values:
            state_values_tensor = torch.empty(0, dtype=torch.float32)
        else:
            # Ensure all state_values are at least 1D for stacking, then squeeze if they were (1,)
            state_values_tensor = torch.stack([sv.float().reshape(-1) for sv in self.state_values]).squeeze()

        is_terminals = torch.tensor(self.is_terminals, dtype=torch.bool)
        
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - is_terminals[t].float() 
                next_value = last_value # last_value is a scalar tensor
            else:
                next_non_terminal = 1.0 - is_terminals[t+1].float()
                next_value = state_values_tensor[t+1] # state_values_tensor is (num_steps,)
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - state_values_tensor[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + state_values_tensor
        return returns, advantages

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.1):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Output action means in [-1, 1] range, to be scaled later if needed
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std_init))

    def forward(self, state):
        raise NotImplementedError # Use act and evaluate methods

    def act(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(axis=-1) # Sum across action dimensions
        state_value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(), dist_entropy

# New CNNActorCritic model
class CNNActorCritic(nn.Module):
    def __init__(self, vector_input_dim, action_dim, action_std_init=0.1, image_channels=1):
        super(CNNActorCritic, self).__init__()
        self.action_dim = action_dim

        # CNN for image processing (200x200 image)
        self.cnn_base = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2), # Output: 16x100x100
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 16x50x50
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # Output: 32x25x25
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32x12x12
            nn.Flatten()
        )
        
        # Calculate flattened CNN output size dynamically
        with torch.no_grad():
            dummy_img = torch.zeros(1, image_channels, 200, 200)
            cnn_flat_dim = self.cnn_base(dummy_img).shape[1] # Should be 32 * 12 * 12 = 4608

        self.image_feature_processor = nn.Sequential(
            self.cnn_base, # Re-use the cnn_base defined above
            nn.Linear(cnn_flat_dim, 256),
            nn.ReLU()
        )

        # MLP for vector input
        self.vector_feature_processor = nn.Sequential(
            nn.Linear(vector_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        combined_features_dim = 256 + 64 # From image_feature_processor and vector_feature_processor
        
        # Shared layers after concatenation
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU()
        )

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh() # Output action means in [-1, 1] range
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(128, 1)
        )
        
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std_init))

    def forward_features(self, image_input, vector_input):
        # image_input: (N, C, 200, 200)
        # vector_input: (N, vector_input_dim)
        img_features = self.image_feature_processor(image_input) # (N, 256)
        vec_features = self.vector_feature_processor(vector_input) # (N, 64)
        combined = torch.cat((img_features, vec_features), dim=1) # (N, 320)
        shared_out = self.shared_layers(combined) # (N, 128)
        return shared_out

    def act(self, image_input, vector_input):
        shared_features = self.forward_features(image_input, vector_input)
        
        action_mean = self.actor_head(shared_features)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(axis=-1, keepdim=True) # Sum across action dimensions, keep batch dim
        
        state_value = self.critic_head(shared_features) # (N, 1)
        
        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, image_input, vector_input, action):
        shared_features = self.forward_features(image_input, vector_input)
        
        action_mean = self.actor_head(shared_features)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(axis=-1) # Sum across action dimensions -> (N,)
        dist_entropy = dist.entropy().sum(axis=-1) # Sum across action dimensions -> (N,)
        state_values = self.critic_head(shared_features).squeeze(-1) # Squeeze to (N,)
        
        return action_logprobs, state_values, dist_entropy

class PPOTrainer:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, K_epochs, eps_clip, 
                 gae_lambda=0.95, action_std_init=0.1, device='cpu', model_type='mlp', image_channels=1):
        self.device = torch.device(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.model_type = model_type

        self.buffer = RolloutBuffer()

        if self.model_type == 'mlp':
            self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
            self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
            optimizer_params = [
                {'params': self.policy.actor.parameters(), 'lr': actor_lr},
                {'params': self.policy.critic.parameters(), 'lr': critic_lr},
                {'params': [self.policy.action_log_std], 'lr': actor_lr * 0.1} # Ensure action_log_std is in a list
            ]
        elif self.model_type == 'cnn':
            # For CNN model, state_dim is interpreted as vector_input_dim
            self.policy = CNNActorCritic(vector_input_dim=state_dim, 
                                         action_dim=action_dim, 
                                         action_std_init=action_std_init,
                                         image_channels=image_channels).to(self.device)
            self.policy_old = CNNActorCritic(vector_input_dim=state_dim, 
                                             action_dim=action_dim, 
                                             action_std_init=action_std_init,
                                             image_channels=image_channels).to(self.device)
            
            optimizer_params = [
                {'params': self.policy.image_feature_processor.parameters(), 'lr': actor_lr},
                {'params': self.policy.vector_feature_processor.parameters(), 'lr': actor_lr},
                {'params': self.policy.shared_layers.parameters(), 'lr': actor_lr},
                {'params': self.policy.actor_head.parameters(), 'lr': actor_lr},
                {'params': self.policy.critic_head.parameters(), 'lr': critic_lr},
                {'params': [self.policy.action_log_std], 'lr': actor_lr * 0.1}
            ]
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'mlp' or 'cnn'.")
        
        self.optimizer = torch.optim.Adam(optimizer_params)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            if self.model_type == 'mlp':
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                action, action_logprob, state_val = self.policy_old.act(state_tensor)
                state_to_buffer = state_tensor.cpu() # state_tensor is (1, state_dim)
            elif self.model_type == 'cnn':
                if not (isinstance(state, tuple) and len(state) == 2):
                    raise ValueError("CNN model expects state as a tuple (image_tensor, vector_tensor)")
                image_input, vector_input = state # Each is (1, C, H, W) and (1, vec_dim) respectively
                
                image_tensor_dev = image_input.to(self.device)
                vector_tensor_dev = vector_input.to(self.device)

                # Ensure batch dimension (get_obs in RLSnake already adds it)
                # if image_tensor_dev.ndim == 3: image_tensor_dev = image_tensor_dev.unsqueeze(0)
                # if vector_tensor_dev.ndim == 1: vector_tensor_dev = vector_tensor_dev.unsqueeze(0)
                
                action, action_logprob, state_val = self.policy_old.act(image_tensor_dev, vector_tensor_dev)
                # Store CPU tensors in buffer. image_input/vector_input are from get_obs, already on CPU & batched.
                state_to_buffer = (image_input.cpu(), vector_input.cpu())
            else:
                 raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # action: (1, action_dim), action_logprob: (1,1) or (1,), state_val: (1,1) or (1,)
        # Squeeze state_val to scalar tensor for buffer, action/logprob will be squeezed later during batching if needed
        self.buffer.add_pending_transition(state_to_buffer, action.cpu(), action_logprob.cpu(), state_val.cpu().squeeze()) 
        return action.cpu().numpy().squeeze(0) # Return numpy array (action_dim,) for environment

    def add_reward_and_done(self, reward, done):
        # Call this after env.step() to update the last transition in the buffer
        self.buffer.update_last_transition(reward, done)

    def update(self, last_value_of_episode):
        if not isinstance(last_value_of_episode, torch.Tensor):
            last_value_tensor = torch.tensor(last_value_of_episode, dtype=torch.float32)
        else:
            last_value_tensor = last_value_of_episode.float()

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=last_value_tensor.squeeze(),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Prepare batch data
        if self.model_type == 'mlp':
            # states are stored as (1, state_dim) tensors in buffer.states
            old_states_batched = torch.stack([s for s in self.buffer.states]).to(self.device).detach().squeeze(1) 
            if old_states_batched.ndim == 1 and len(self.buffer.states) == 1: # case of single step in buffer, stack might not give (1, dim)
                 old_states_batched = old_states_batched.unsqueeze(0)
        elif self.model_type == 'cnn':
            # self.buffer.states contains list of (image_cpu (1,C,H,W), vector_cpu (1,vec_dim)) tuples
            if not self.buffer.states:
                 raise ValueError("Buffer is empty, cannot update CNN model.")
            if not (isinstance(self.buffer.states[0], tuple) and len(self.buffer.states[0]) == 2):
                raise ValueError("CNN model expects buffer states to be (image, vector) tuples.")

            old_images_batched = torch.cat([s[0] for s in self.buffer.states], dim=0).to(self.device).detach()
            old_vectors_batched = torch.cat([s[1] for s in self.buffer.states], dim=0).to(self.device).detach()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        # Actions and Logprobs are stored as (1, dim) or (1,) tensors. Stack and squeeze.
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        if old_actions.ndim == 3 and old_actions.shape[1] == 1: # (N, 1, action_dim) -> (N, action_dim)
            old_actions = old_actions.squeeze(1)
        elif old_actions.ndim == 2 and old_actions.shape[0] == len(self.buffer.states) and old_actions.shape[1] == self.policy.action_dim: # (N, action_dim)
            pass # Correctly shaped
        # Handle case where action_dim is 1, squeeze might make it (N,). Reshape to (N,1)
        if old_actions.ndim == 1 and self.policy.action_dim == 1:
             old_actions = old_actions.unsqueeze(-1)

        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()
        if old_logprobs.ndim == 2 and old_logprobs.shape[1] == 1: # (N, 1) -> (N,)
            old_logprobs = old_logprobs.squeeze(-1)
        elif old_logprobs.ndim == 1 and old_logprobs.shape[0] == len(self.buffer.states):
            pass # Correctly shaped (N,)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            if self.model_type == 'mlp':
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batched, old_actions)
            elif self.model_type == 'cnn':
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_images_batched, old_vectors_batched, old_actions)
            else:
                raise ValueError(f"Unknown model_type during policy evaluation: {self.model_type}")

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns) 
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        print(f"PPO model saved at {checkpoint_path}")

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        print(f"PPO model loaded from {checkpoint_path}")

    def get_value(self, state):
        with torch.no_grad():
            if self.model_type == 'mlp':
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                if state_tensor.ndim == 1: 
                    state_tensor = state_tensor.unsqueeze(0)
                _, _, state_value_full = self.policy_old.act(state_tensor) 
            elif self.model_type == 'cnn':
                if not (isinstance(state, tuple) and len(state) == 2):
                    raise ValueError("CNN model expects state as a tuple (image_tensor, vector_tensor) for get_value")
                image_input, vector_input = state # Each is (1,C,H,W) and (1,vec_dim)

                image_tensor_dev = image_input.to(self.device)
                vector_tensor_dev = vector_input.to(self.device)

                # Ensure batch dimension (get_obs in RLSnake already adds it)
                # if image_tensor_dev.ndim == 3: image_tensor_dev = image_tensor_dev.unsqueeze(0)
                # if vector_tensor_dev.ndim == 1: vector_tensor_dev = vector_tensor_dev.unsqueeze(0)

                _, _, state_value_full = self.policy_old.act(image_tensor_dev, vector_tensor_dev)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            
        return state_value_full.cpu().squeeze().item()
