import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import wandb

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)


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
            nn.Tanh(),  # Output action means in [-1, 1] range, to be scaled later if needed
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.action_log_std = nn.Parameter(
            torch.ones(1, action_dim) * np.log(action_std_init)
        )

    def forward(self, state):
        raise NotImplementedError  # Use act and evaluate methods

    def act(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(
            axis=-1
        )  # Sum across action dimensions
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


class BCBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.gt_actions = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.gt_actions[:]

    def add_transition(self, state, action, gt_action):
        self.states.append(state)
        self.actions.append(action)
        self.gt_actions.append(gt_action)


class BCTrainer:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr,
        critic_lr,
        gamma,
        K_epochs,
        eps_clip,
        gae_lambda=0.95,
        action_std_init=0.1,
        device="cpu",
        model_type="mlp",
        image_channels=1,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.model_type = model_type
        self.get_gt_action = None  # Will be set externally

        self.buffer = BCBuffer()

        if self.model_type == "mlp":
            self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(
                self.device
            )
            optimizer_params = [
                {"params": self.policy.actor.parameters(), "lr": actor_lr},
                {"params": [self.policy.action_log_std], "lr": actor_lr * 0.1},
            ]
        elif self.model_type == "cnn":
            # For CNN model, state_dim is interpreted as vector_input_dim
            from ppo_trainer import CNNActorCritic

            self.policy = CNNActorCritic(
                vector_input_dim=state_dim,
                action_dim=action_dim,
                action_std_init=action_std_init,
                image_channels=image_channels,
            ).to(self.device)

            optimizer_params = [
                {
                    "params": self.policy.image_feature_processor.parameters(),
                    "lr": actor_lr,
                },
                {
                    "params": self.policy.vector_feature_processor.parameters(),
                    "lr": actor_lr,
                },
                {"params": self.policy.shared_layers.parameters(), "lr": actor_lr},
                {"params": self.policy.actor_head.parameters(), "lr": actor_lr},
                {"params": [self.policy.action_log_std], "lr": actor_lr * 0.1},
            ]
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. Choose 'mlp' or 'cnn'."
            )

        self.optimizer = torch.optim.Adam(optimizer_params)

        # Store current state for GT action collection
        self.current_state = None

    def set_gt_action_fn(self, gt_action_fn):
        """Set the ground truth action function"""
        self.get_gt_action = gt_action_fn

    def select_action(self, state):
        """Select action using current policy and store state for GT collection"""
        with torch.no_grad():
            if self.model_type == "mlp":
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                action, action_logprob, state_val = self.policy.act(state_tensor)
                self.current_state = state_tensor.cpu()
            elif self.model_type == "cnn":
                if not (isinstance(state, tuple) and len(state) == 2):
                    raise ValueError(
                        "CNN model expects state as a tuple (image_tensor, vector_tensor)"
                    )
                image_input, vector_input = state

                image_tensor_dev = image_input.to(self.device)
                vector_tensor_dev = vector_input.to(self.device)

                action, action_logprob, state_val = self.policy.act(
                    image_tensor_dev, vector_tensor_dev
                )
                self.current_state = (image_input.cpu(), vector_input.cpu())
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        # Store the predicted action
        self.current_action = action.cpu()
        return action.cpu().numpy().squeeze(0)

    def add_reward_and_done(self, reward, done):
        """Compatibility method - collect GT action and add to buffer"""
        if self.get_gt_action is None:
            raise ValueError(
                "Ground truth action function not set. Call set_gt_action_fn() first."
            )

        if self.current_state is None:
            return

        # Get ground truth action for the current state
        gt_action = self.get_gt_action(self.current_state)

        # Convert gt_action to tensor if needed
        if not isinstance(gt_action, torch.Tensor):
            gt_action = torch.FloatTensor(gt_action)
        if gt_action.ndim == 1:
            gt_action = gt_action.unsqueeze(0)

        # Add transition to buffer
        self.buffer.add_transition(self.current_state, self.current_action, gt_action)

    def update(self, last_value_of_episode=None):
        """Train the actor to match ground truth actions"""
        if len(self.buffer.states) == 0:
            return

        # Prepare batch data
        if self.model_type == "mlp":
            states_batched = (
                torch.stack([s for s in self.buffer.states]).to(self.device).squeeze(1)
            )
            if states_batched.ndim == 1 and len(self.buffer.states) == 1:
                states_batched = states_batched.unsqueeze(0)
        elif self.model_type == "cnn":
            if not self.buffer.states:
                return
            if not (
                isinstance(self.buffer.states[0], tuple)
                and len(self.buffer.states[0]) == 2
            ):
                raise ValueError(
                    "CNN model expects buffer states to be (image, vector) tuples."
                )

            images_batched = torch.cat([s[0] for s in self.buffer.states], dim=0).to(
                self.device
            )
            vectors_batched = torch.cat([s[1] for s in self.buffer.states], dim=0).to(
                self.device
            )

        # Ground truth actions
        gt_actions = torch.stack(self.buffer.gt_actions).to(self.device)
        if gt_actions.ndim == 3 and gt_actions.shape[1] == 1:
            gt_actions = gt_actions.squeeze(1)
        gt_actions = gt_actions.reshape(gt_actions.shape[0], -1)

        avg_loss = 0.0
        # Train for K_epochs
        for _ in range(self.K_epochs):
            if self.model_type == "mlp":
                action_mean = self.policy.actor(states_batched)
            elif self.model_type == "cnn":
                shared_features = self.policy.forward_features(
                    images_batched, vectors_batched
                )
                action_mean = self.policy.actor_head(shared_features)

            # Behavioral cloning loss - MSE between predicted and ground truth actions
            bc_loss = F.mse_loss(action_mean, gt_actions)
            avg_loss += bc_loss.item()

            self.optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.buffer.clear()
        return avg_loss / self.K_epochs

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
        print(f"BC model saved at {checkpoint_path}")

    def load(self, checkpoint_path):
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        print(f"BC model loaded from {checkpoint_path}")

    def get_value(self, state):
        """Get state value - for compatibility with RL interface"""
        with torch.no_grad():
            if self.model_type == "mlp":
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                _, _, state_value = self.policy.act(state_tensor)
            elif self.model_type == "cnn":
                if not (isinstance(state, tuple) and len(state) == 2):
                    raise ValueError(
                        "CNN model expects state as a tuple (image_tensor, vector_tensor) for get_value"
                    )
                image_input, vector_input = state

                image_tensor_dev = image_input.to(self.device)
                vector_tensor_dev = vector_input.to(self.device)

                _, _, state_value = self.policy.act(image_tensor_dev, vector_tensor_dev)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        return state_value.cpu().squeeze().item()
