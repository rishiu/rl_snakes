"""
This file implements PPO algorithm.

-----
Taken from: 

CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
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
        self.is_terminals = [] # Stores done flags

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, logprob, reward, state_value, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(is_terminal)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        state_values = torch.tensor(self.state_values, dtype=torch.float32)
        is_terminals = torch.tensor(self.is_terminals, dtype=torch.bool)
        
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - is_terminals[t].float() # Should be 1 if last step of episode is not truly terminal
                next_value = last_value # Value of the state after the last collected step
            else:
                next_non_terminal = 1.0 - is_terminals[t+1].float()
                next_value = state_values[t+1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - state_values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + state_values
        return returns, advantages

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.1):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Output action means in [-1, 1] range, to be scaled later if needed
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
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

class PPOTrainer:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, K_epochs, eps_clip, gae_lambda=0.95, action_std_init=0.1, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': critic_lr},
            {'params': self.policy.action_log_std, 'lr': actor_lr * 0.1} # Smaller LR for std
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state_tensor)
        
        # Store transition in buffer
        # Note: state should be stored as numpy/list, action as tensor for batching later
        self.buffer.add(state, action.cpu(), action_logprob.cpu(), None, state_val.cpu(), None) # Rewards and dones added later
        return action.cpu().numpy() # Return numpy array for environment

    def add_reward_and_done(self, reward, done):
        # Call this after env.step()
        if self.buffer.rewards and self.buffer.rewards[-1] is None: # if last reward is placeholder
             self.buffer.rewards[-1] = reward
             self.buffer.is_terminals[-1] = done
        else: # This case should not happen if select_action is always followed by add_reward_and_done
            print("Warning: Adding reward/done to a non-placeholder or empty buffer entry.")


    def update(self, last_value_of_episode):
        # Monte Carlo estimation of returns and GAE
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=torch.tensor(last_value_of_episode, dtype=torch.float32).to(self.device),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Optimize policy for K epochs
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns.to(self.device))
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean() # Add entropy bonus

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
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
            state_tensor = torch.FloatTensor(state).to(self.device)
            # The critic is part of policy_old.act, but we need only the value for the last step
            # So, let's call the critic directly.
            state_value = self.policy_old.critic(state_tensor)
        return state_value.cpu().item()


if __name__ == '__main__':
    # You can run the script here to see if your PPO is implemented correctly

    from utils import register_metadrive
    from envs import make_envs

    env_name = "MetaDrive-Tut-Easy-v0"

    register_metadrive()

    class FakeConfig(PPOConfig):
        def __init__(self):
            super(FakeConfig, self).__init__()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_envs = 1
            self.num_steps = 200
            self.gamma = 0.99
            self.lr = 5e-4
            self.grad_norm_max = 10.0
            self.value_loss_weight = 1.0
            self.entropy_loss_weight = 0.0

    env = make_envs("CartPole-v0", num_envs=3)
    trainer = PPOTrainer(env, FakeConfig())
    obs = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert actions.shape == (1, 1), actions.shape
    assert values.shape == (1, 1), values.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert actions.shape == (3, 1), actions.shape
    assert values.shape == (3, 1), values.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    print("Base trainer discrete case test passed!")
    env.close()

    # ===== Continuous case =====
    env = make_envs("BipedalWalker-v3", asynchronous=False, num_envs=3)
    trainer = PPOTrainer(env, FakeConfig())
    obs = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert values.shape == (1, 1), values.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert values.shape == (3, 1), values.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    print("Base trainer continuous case test passed!")
    env.close()
