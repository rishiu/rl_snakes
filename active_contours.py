import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ext_energy.gradient_energy import GradientExternalEnergy
# from data_utils import build_dataset_setting_fn
from utils import (
    ma,
    create_circle_points,
    create_ellipse_points,
    bilinear_interpolate,
    calculate_normals,
)
from viz import (
    visualize_snake_evolution,
    visualize_gradient_fields,
    plot_snake_on_image,
    visualize_snake_evolution_with_energy_rewards,
)
from mask_utils import (
    create_multi_circle_mask,
    create_centered_rectangular_mask,
    create_circular_mask,
    create_triangle_mask,
    create_star_mask,
    create_elliptical_mask,
)
from RL.ppo_trainer import PPOTrainer
from RL.bc_trainer import BCTrainer
import os
import wandb
import matplotlib.pyplot as plt
import cv2


class RLSnake:
    def __init__(
        self,
        initial_points,
        external_energy_fn,
        setting_fn,
        run_name,
        update_freq=1,
        save_freq=20,
        alpha=0.01,
        beta=0.01,
        gamma=0.01,
        model_type="mlp",
        image_channels=1,
        obs_type="trad",
    ):
        self.num_control_points = initial_points.shape[1]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_points = initial_points
        self.initial_points = initial_points.copy()
        self.ext_energy = external_energy_fn
        self.curr_img = None
        self.setting_fn = setting_fn
        self.update_freq = update_freq
        self.save_freq = save_freq
        self.run_name = run_name
        self.model_type = model_type
        self.image_channels = image_channels
        self.obs_type = obs_type

        os.makedirs(f"runs/{run_name}/models", exist_ok=True)
        self.run = wandb.init(
            project="RLSnake",
            name=run_name,
            config={
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "update_freq": update_freq,
                "save_freq": save_freq,
                "run_name": run_name,
                "num_control_points": self.num_control_points,
                "model_type": self.model_type,
            },
        )

        self.vector_input_dim = self.num_control_points * 2 + (
            32 * 32
        )  # self.get_obs(self.obs_type).shape[0]
        self.output_dim = self.num_control_points * 2

        self.trainer = PPOTrainer(self.vector_input_dim, self.output_dim, \
                                  actor_lr=5e-5, critic_lr=5e-5, \
                                  gamma=0.99, K_epochs=self.update_freq, eps_clip=0.2, \
                                  action_std_init=0.1,
                                  model_type=self.model_type,
                                  image_channels=self.image_channels)

        # self.trainer = BCTrainer(self.vector_input_dim, self.output_dim, \
        #                          actor_lr=5e-5, critic_lr=5e-5, \
        #                          gamma=0.99, K_epochs=self.update_freq, eps_clip=0.2, \
        #                          action_std_init=0.1,
        #                          model_type=self.model_type,
        #                          image_channels=self.image_channels)
        

        A = Snake.setup_A(self.num_control_points, 0.05, 0.1)
        A_ = A + 1.0 * np.eye(self.num_control_points)

        def gt_action_fn(obs):
            snake_points = np.array(
                obs[0, : self.num_control_points * 2].reshape(2, -1)
            )

            f_x, f_y = self.ext_energy(snake_points)
            dext = np.array([f_x, f_y])

            y = 1.0 * snake_points - dext

            new_snake_points = np.zeros_like(snake_points)
            new_snake_points[0, :] = np.linalg.solve(A_, y[0, :])
            new_snake_points[1, :] = np.linalg.solve(A_, y[1, :])
            return new_snake_points - snake_points
        
        # self.trainer.set_gt_action_fn(gt_action_fn)

    def compute_internal_energy(self):
        temp_points = self.current_points.copy()
        diff = np.power(temp_points - np.roll(temp_points, 1, axis=1), 2)

        diff_ = np.power(diff - np.roll(diff, 1, axis=1), 2)

        return self.alpha * np.sum(diff) + self.beta * np.sum(diff_)

    def compute_external_energy(self):
        temp_points = self.current_points.copy()
        return 2 * self.ext_energy.get_energy(temp_points)

    def sample_image(self):
        return bilinear_interpolate(self.curr_img, self.current_points)

    def get_image_roi(self, points, target_size=(32, 32)):
        min_x = int(np.min(points[0, :]))
        max_x = int(np.max(points[0, :]))
        min_y = int(np.min(points[1, :]))
        max_y = int(np.max(points[1, :]))

        roi = self.curr_img[min_y:max_y, min_x:max_x]

        roi_normalized = (roi / 255.0 - 0.5) * 2.0

        roi_normalized = cv2.resize(
            roi_normalized, target_size, interpolation=cv2.INTER_LINEAR
        )

        return torch.tensor(roi_normalized.flatten(), dtype=torch.float32)

    def get_com_obs(self):
        if self.current_points is None:
            raise ValueError(
                "current_points is not initialized before calling get_com_obs."
            )
        center = np.mean(self.current_points, axis=1, keepdims=True)

        relative_vectors = self.current_points - center

        com_features_tensor = torch.tensor(
            relative_vectors.flatten(), dtype=torch.float32
        )

        return com_features_tensor

    def get_normal(self):
        return calculate_normals(self.current_points)

    def get_obs(self, obs_type="com,roi"):
        obs_types = obs_type.split(",")
        obs = []
        if "trad" in obs_types:
            obs.append(self.get_trad_obs())
        if "com" in obs_types:
            obs.append(self.get_com_obs())
        if "roi" in obs_types:
            obs.append(self.get_image_roi(self.current_points, target_size=(32, 32)))
        return torch.cat(obs, dim=0)  # .unsqueeze(0).unsqueeze(0)

    def get_trad_obs(self):
        points_normalized = (
            (
                torch.tensor(self.current_points.copy().flatten(), dtype=torch.float32)
                / 200.0
            )
            - 0.5
        ) * 2.0

        img_vals_at_points_normalized = (
            torch.tensor(self.sample_image(), dtype=torch.float32) / 255.0 - 0.5
        ) * 2.0

        vector_features = torch.cat(
            [points_normalized, img_vals_at_points_normalized], dim=0
        )

        if self.model_type == "mlp":
            return vector_features.unsqueeze(0)
        elif self.model_type == "cnn":
            if self.curr_img is None:
                raise ValueError(
                    "Current image (self.curr_img) is not set for CNN model."
                )

            image_tensor = torch.tensor(self.curr_img, dtype=torch.float32) / 255.0
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(0)

            vector_feature_tensor = vector_features.unsqueeze(0)

            return image_tensor, vector_feature_tensor
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type} in get_obs")

    def run_setting(self, setting_fn, setting_idx, num_steps=200):
        img, gt_points = setting_fn()

        wandb.log({"img": wandb.Image(img)}, step=setting_idx)

        self.curr_img = img
        self.ext_energy = external_energy_fn(img)

        rewards = []
        internal_energies_over_time = []
        external_energies_over_time = []

        reward_by_time = np.zeros(num_steps)

        last_reward = 0.1 * (
            self.compute_external_energy()
            + np.exp(-1.5 * self.compute_internal_energy())
        )

        snake_evolution = []

        for step in range(num_steps):
            snake_evolution.append(self.current_points.copy())
            obs = self.get_obs(self.obs_type)
            offset = self.trainer.select_action(obs).reshape(2, self.num_control_points)

            # normal = self.get_normal()[0]
            # offset = offset * normal

            self.current_points = self.current_points + offset
            self.current_points[0, :] = np.clip(
                self.current_points[0, :], 0, self.curr_img.shape[0] - 1
            )
            self.current_points[1, :] = np.clip(
                self.current_points[1, :], 0, self.curr_img.shape[1] - 1
            )

            internal_energy = self.compute_internal_energy()
            external_energy = self.compute_external_energy()
            internal_energies_over_time.append(internal_energy)
            external_energies_over_time.append(external_energy)

            if gt_points is not None:
                reward = (
                    np.exp(-0.1 * np.mean(np.abs(self.current_points - gt_points)))
                    + external_energy
                    - internal_energy
                )
                reward_by_time[step] = reward
            else:
                reward = (
                    0.1 * (external_energy + np.exp(-0.1 * internal_energy))
                ) - last_reward
                last_reward = reward
                reward_by_time[step] = reward

            if internal_energy > 0.1:
                reward = -5.0
                done = True
            elif np.abs(external_energy - internal_energy) < 1e-5:
                reward = 5.0
                done = True
            elif gt_points is not None:
                done = np.mean(np.abs(self.current_points - gt_points)) < 0.1
            else:
                done = step == num_steps - 1

            rewards.append(reward)

            self.trainer.add_reward_and_done(reward, done)

            if done:
                break

        # if setting_idx > 20:
        #     visualize_snake_evolution_with_energy_rewards(
        #         self.curr_img,
        #         snake_evolution,
        #         rewards_list=rewards,
        #         internal_energy_list=internal_energies_over_time,
        #         external_energy_list=external_energies_over_time,
        #         title="RL Snake Evolution with Metrics"
        #     )

        self.last_value = self.trainer.get_value(self.get_obs(self.obs_type))

        snake_fig = plot_snake_on_image(self.curr_img, self.current_points)
        wandb.log({"snake": snake_fig}, step=setting_idx)
        plt.close(snake_fig)

        self.current_points = self.initial_points.copy()

        return rewards, reward_by_time

    def optimize(self, num_settings=100, num_steps=200):
        avg_rewards_by_time = np.zeros(num_steps)
        for i in tqdm(range(num_settings)):
            rewards, reward_by_time = self.run_setting(self.setting_fn, i, num_steps)
            wandb.log({"reward": np.mean(rewards), "final_reward": rewards[-1]}, step=i)
            avg_rewards_by_time += reward_by_time

            if i % self.update_freq == 0:
                bc_loss = self.trainer.update(self.last_value)
                wandb.log({"bc_loss": bc_loss}, step=i)
            if i % self.save_freq == 0:
                self.trainer.save(f"runs/{self.run_name}/models/snake_model_{i}.pth")
                table = wandb.Table(
                    data=[
                        [j, avg_rewards_by_time[j] / self.update_freq]
                        for j in range(num_steps)
                    ],
                    columns=["step", "avg_reward_by_time"],
                )
                wandb.log(
                    {
                        f"avg_reward_by_time_{i}": wandb.plot.line(
                            table,
                            "step",
                            "avg_reward_by_time",
                            title=f"Avg Reward by Time {i}",
                        )
                    }
                )
                avg_rewards_by_time = np.zeros(num_steps)

    def evaluate(self, checkpoint_path, setting_fn=None, num_episodes=1, num_steps=200, log_to_wandb=False):
        """
        Evaluate the trained model from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
            setting_fn (callable, optional): Function to generate evaluation settings. If None, uses self.setting_fn
            num_episodes (int): Number of evaluation episodes to run (default: 1)
            num_steps (int): Maximum steps per episode
            log_to_wandb (bool): Whether to log evaluation results to wandb
            
        Returns:
            tuple: (eval_metrics dict, final_snake_image) where final_snake_image is the visualization of the final snake
        """
        # Load the model from checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        self.trainer.load(checkpoint_path)
        
        # Use provided setting function or default to the instance's setting function
        eval_setting_fn = setting_fn if setting_fn is not None else self.setting_fn
        
        # Store evaluation metrics
        episode_rewards = []
        final_rewards = []
        episode_lengths = []
        internal_energies = []
        external_energies = []
        final_snake_image = None
        
        print(f"Running evaluation for {num_episodes} episodes...")
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            # Reset snake to initial position
            self.current_points = self.initial_points.copy()
            
            # Get new setting
            img, gt_points = eval_setting_fn()
            self.curr_img = img
            self.ext_energy = external_energy_fn(img)
            
            episode_reward = 0
            step_count = 0
            internal_energy_episode = []
            external_energy_episode = []
            
            if log_to_wandb:
                wandb.log({"eval_img": wandb.Image(img)}, step=episode)
            
            # Run episode
            for step in range(num_steps):
                obs = self.get_obs(self.obs_type)
                offset = self.trainer.select_action(obs).reshape(2, self.num_control_points)
                
                self.current_points = self.current_points + offset 
                self.current_points[0,:] = np.clip(self.current_points[0,:], 0, self.curr_img.shape[0] - 1)
                self.current_points[1,:] = np.clip(self.current_points[1,:], 0, self.curr_img.shape[1] - 1)
                
                internal_energy = self.compute_internal_energy()
                external_energy = self.compute_external_energy()
                internal_energy_episode.append(internal_energy)
                external_energy_episode.append(external_energy)
                
                # Calculate reward (same logic as in run_setting)
                if gt_points is not None:
                    reward = np.exp(-0.1 * np.mean(np.abs(self.current_points - gt_points))) + external_energy - internal_energy
                else:
                    reward = 0.1 * (external_energy + np.exp(-0.1 * internal_energy))
                
                episode_reward += reward
                step_count += 1
                
                # Check termination conditions
                if internal_energy > 0.1:
                    break
                elif np.abs(external_energy - internal_energy) < 1e-5:
                    break
                elif gt_points is not None and np.mean(np.abs(self.current_points - gt_points)) < 0.1:
                    break
            
            episode_rewards.append(episode_reward)
            final_rewards.append(reward)
            episode_lengths.append(step_count)
            internal_energies.extend(internal_energy_episode)
            external_energies.extend(external_energy_episode)
            
            # Generate final snake image (for the last episode or if only 1 episode)
            snake_fig = plot_snake_on_image(self.curr_img, self.current_points)
            if episode == num_episodes - 1:  # Save the final episode's image
                final_snake_image = snake_fig
            
            if log_to_wandb:
                wandb.log({
                    "eval_snake": snake_fig,
                    "eval_episode_reward": episode_reward,
                    "eval_final_reward": reward,
                    "eval_episode_length": step_count
                }, step=episode)
            
            # Only close the figure if it's not the final one we want to return
            if episode != num_episodes - 1:
                plt.close(snake_fig)
        
        # Calculate evaluation metrics
        eval_metrics = {
            "mean_episode_reward": np.mean(episode_rewards),
            "std_episode_reward": np.std(episode_rewards),
            "mean_final_reward": np.mean(final_rewards),
            "std_final_reward": np.std(final_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "mean_internal_energy": np.mean(internal_energies),
            "mean_external_energy": np.mean(external_energies),
            "episode_rewards": episode_rewards,
            "final_rewards": final_rewards,
            "episode_lengths": episode_lengths
        }
        
        # Log summary metrics
        if log_to_wandb:
            wandb.log({
                "eval_mean_episode_reward": eval_metrics["mean_episode_reward"],
                "eval_std_episode_reward": eval_metrics["std_episode_reward"],
                "eval_mean_final_reward": eval_metrics["mean_final_reward"],
                "eval_mean_episode_length": eval_metrics["mean_episode_length"],
                "eval_mean_internal_energy": eval_metrics["mean_internal_energy"],
                "eval_mean_external_energy": eval_metrics["mean_external_energy"]
            })
        
        print(f"Evaluation completed!")
        print(f"Mean episode reward: {eval_metrics['mean_episode_reward']:.4f} ± {eval_metrics['std_episode_reward']:.4f}")
        print(f"Mean final reward: {eval_metrics['mean_final_reward']:.4f} ± {eval_metrics['std_final_reward']:.4f}")
        print(f"Mean episode length: {eval_metrics['mean_episode_length']:.2f} ± {eval_metrics['std_episode_length']:.2f}")
        
        return eval_metrics, final_snake_image

    def evaluate_on_all_settings(self, checkpoint_path, output_dir="output_ppo", num_steps=200, log_to_wandb=False):
        """
        Evaluate the trained model on all available setting functions and save results.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
            output_dir (str): Directory to save evaluation images (default: "output")
            num_steps (int): Maximum steps per episode
            log_to_wandb (bool): Whether to log evaluation results to wandb
            
        Returns:
            dict: Dictionary with setting names as keys and evaluation results as values
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define all setting functions
        img_height, img_width = 200, 200
        
        def rect_setting_fn():
            img = create_centered_rectangular_mask(img_height, img_width, rect_height=img_height//4, rect_width=img_width//4)
            return img, None
        
        def circle_setting_fn():
            img = create_circular_mask(img_height, img_width, radius=img_width//4)
            return img, None
        
        def multicircle_setting_fn():
            img = create_multi_circle_mask(img_height, img_width, num_circles=5, radius_range=(10, 25))
            return img, None
        
        def triangle_setting_fn():
            img = create_triangle_mask(
                img_height,
                img_width,
                center_x=img_width // 2,
                center_y=img_height // 2,
                base=img_width // 3,
                triangle_height=img_height // 3,
                orientation="up",
            )
            return img, None

        def star_setting_fn():
            img = create_star_mask(
                img_height,
                img_width,
                center_x=img_width // 2,
                center_y=img_height // 2,
                outer_radius=img_width // 4,
                inner_radius=img_width // 8,
                num_points=5,
            )
            return img, None

        def ellipse_setting_fn():
            img = create_elliptical_mask(
                img_height,
                img_width,
                center_x=img_width // 2,
                center_y=img_height // 2,
                radius_x=img_width // 3,
                radius_y=img_height // 4,
                rotation_angle_rad=0,
            )
            return img, None
        
        def test_eagle_image():
            img = np.array(Image.open("test_images/eagle.jpeg").convert('L'))
            return img, None
        
        # Dictionary of setting functions
        setting_functions = {
            "rectangle": rect_setting_fn,
            "circle": circle_setting_fn,
            "multicircle": multicircle_setting_fn,
            "triangle": triangle_setting_fn,
            "star": star_setting_fn,
            "ellipse": ellipse_setting_fn,
            "eagle": test_eagle_image
        }
        
        print(f"Evaluating checkpoint: {checkpoint_path}")
        print(f"Testing on {len(setting_functions)} different settings...")
        print(f"Results will be saved to: {output_dir}/")
        
        # Store results
        all_results = {}
        
        # Load the model once
        print(f"Loading model from checkpoint: {checkpoint_path}")
        self.trainer.load(checkpoint_path)
        
        # Evaluate on each setting
        for setting_name, setting_fn in setting_functions.items():
            print(f"\nEvaluating on {setting_name} setting...")
            
            try:
                # Run evaluation
                eval_metrics, final_snake_image = self.evaluate(
                    checkpoint_path=checkpoint_path,
                    setting_fn=setting_fn,
                    num_episodes=1,
                    num_steps=num_steps,
                    log_to_wandb=log_to_wandb
                )
                
                # Save the image
                image_path = os.path.join(output_dir, f"{setting_name}_evaluation.png")
                final_snake_image.savefig(image_path, dpi=150, bbox_inches='tight')
                plt.close(final_snake_image)  # Close to free memory
                
                # Store results
                all_results[setting_name] = {
                    "metrics": eval_metrics,
                    "image_path": image_path
                }
                
                print(f"✓ {setting_name}: Reward={eval_metrics['mean_episode_reward']:.4f}, "
                      f"Length={eval_metrics['mean_episode_length']:.1f} steps, "
                      f"Image saved to {image_path}")
                
            except Exception as e:
                print(f"✗ Failed to evaluate {setting_name}: {str(e)}")
                all_results[setting_name] = {
                    "metrics": None,
                    "image_path": None,
                    "error": str(e)
                }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Setting':<15} {'Reward':<10} {'Length':<10} {'Status':<15}")
        print(f"{'-'*60}")
        
        for setting_name, result in all_results.items():
            if result["metrics"] is not None:
                reward = result["metrics"]["mean_episode_reward"]
                length = result["metrics"]["mean_episode_length"]
                status = "✓ Success"
                print(f"{setting_name:<15} {reward:<10.4f} {length:<10.1f} {status:<15}")
            else:
                print(f"{setting_name:<15} {'N/A':<10} {'N/A':<10} {'✗ Failed':<15}")
        
        # Log summary to wandb if enabled
        if log_to_wandb:
            summary_metrics = {}
            for setting_name, result in all_results.items():
                if result["metrics"] is not None:
                    summary_metrics[f"eval_all_{setting_name}_reward"] = result["metrics"]["mean_episode_reward"]
                    summary_metrics[f"eval_all_{setting_name}_length"] = result["metrics"]["mean_episode_length"]
            wandb.log(summary_metrics)
        
        print(f"\nAll evaluation images saved to: {output_dir}/")
        return all_results
class Snake:
    def __init__(
        self, initial_points, external_energy, alpha=0.01, beta=0.01, gamma=0.01
    ):
        if (
            not isinstance(initial_points, np.ndarray)
            or initial_points.ndim != 2
            or initial_points.shape[0] != 2
        ):
            raise ValueError("initial_points must be a NumPy array of shape (2, N)")

        self.num_control_points = initial_points.shape[1]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_points = initial_points
        self.ext_energy = external_energy

    @staticmethod
    def setup_A(N, alphas, betas):
        if np.isscalar(alphas):
            alphas = np.full(N, alphas)
        if np.isscalar(betas):
            betas = np.full(N, betas)
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = (
                ma(alphas, i)
                + ma(alphas, i + 1)
                + 4 * ma(betas, i)
                + ma(betas, i + 1)
                + ma(betas, i - 1)
            )
            A[i, (i - 1) % N] = -ma(alphas, i) - 2 * ma(betas, i) - 2 * ma(betas, i - 1)
            A[i, (i + 1) % N] = (
                -ma(alphas, i + 1) - 2 * ma(betas, i) - 2 * ma(betas, i + 1)
            )
            A[i, (i - 2) % N] = ma(betas, i - 1)
            A[i, (i + 2) % N] = ma(betas, i + 1)
        return A

    def optimize(self, num_iterations=100):
        A = Snake.setup_A(self.num_control_points, self.alpha, self.beta)

        A_ = A + self.gamma * np.eye(self.num_control_points)

        snake_points = self.current_points.copy()

        snake_evolution = []

        for i in tqdm(range(num_iterations)):
            snake_evolution.append(snake_points.copy())

            f_x, f_y = self.ext_energy(snake_points)
            dext = np.array([f_x, f_y])

            y = self.gamma * snake_points - dext

            snake_points[0, :] = np.linalg.solve(A_, y[0, :])
            snake_points[1, :] = np.linalg.solve(A_, y[1, :])

        snake_evolution.append(snake_points.copy())
        self.current_points = snake_points.copy()
        return snake_evolution

def rect_setting_fn():
    img = create_centered_rectangular_mask(img_height, img_width, rect_height=img_height//4, rect_width=img_width//4)
    return img, None

def circle_setting_fn():
    img = create_circular_mask(img_height, img_width, radius=img_width//4)
    return img, None

def setting_fn():
    img = create_multi_circle_mask(img_height, img_width, num_circles=5, radius_range=(10, 25))
    return img, None

def triangle_setting_fn():
    img = create_triangle_mask(
        img_height,
        img_width,
        center_x=img_width // 2,
        center_y=img_height // 2,
        base=img_width // 3,
        triangle_height=img_height // 3,
        orientation="up",
    )
    return img, None

def star_setting_fn():
    img = create_star_mask(
        img_height,
        img_width,
        center_x=img_width // 2,
        center_y=img_height // 2,
        outer_radius=img_width // 4,
        inner_radius=img_width // 8,
        num_points=5,
    )
    return img, None

def ellipse_setting_fn():
    img = create_elliptical_mask(
        img_height,
        img_width,
        center_x=img_width // 2,
        center_y=img_height // 2,
        radius_x=img_width // 3,
        radius_y=img_height // 4,
        rotation_angle_rad=0,
    )
    return img, None

def test_eagle_image():
    img = np.array(Image.open("test_images/eagle.jpeg").convert('L'))
    return img, None

if __name__ == "__main__":
    img = np.array(Image.open("test_images/eagle.jpeg").convert('L'))
    img_height, img_width = img.shape
    num_snake_points = 30
    initial_snake = create_circle_points(num_snake_points, 
                                         center_x=img_width//2, 
                                         center_y=img_height//2, 
                                         radius=img_width//3)

    def external_energy_fn(img):
        return GradientExternalEnergy(img)       

    rl_snake = RLSnake(initial_points=initial_snake, 
                       external_energy_fn=external_energy_fn,
                       setting_fn=test_eagle_image,
                       run_name="rl_snake_eagle_bc_30",
                       update_freq=1,
                       save_freq=500,
                       alpha=5e-7,
                       beta=1e-8,
                       model_type='mlp',
                       obs_type='com,roi')
    
    rl_snake.evaluate_on_all_settings(checkpoint_path="runs/rl_snake_multicircle_ppo_30/models/snake_model_1500.pth",
                                      num_steps=200)

    rl_snake.optimize(num_settings=3000)

# if __name__ == "__main__":
#     img_height, img_width = 200, 200
#     num_snake_points = 30

#     RL = True

    
#     # dataset_setting_fn = build_dataset_setting_fn("./datasets/")
    
#     def external_energy_fn(img):
#         return GradientExternalEnergy(img)
    
#     initial_snake = create_circle_points(num_snake_points, 
#                                          center_x=img_width//2, 
#                                          center_y=img_height//2, 
#                                          radius=img_width//3)
    
#     if RL:
#         rl_snake = RLSnake(initial_points=initial_snake, 
#                        external_energy_fn=external_energy_fn,
#                        setting_fn=setting_fn,
#                        run_name="rl_snake_multicircle_ppo_30",
#                        update_freq=1,
#                        save_freq=500,
#                        alpha=5e-7,
#                        beta=1e-8,
#                        model_type='mlp',
#                        obs_type='com,roi')

#         rl_snake.optimize(num_settings=3000)
        
#         # Example of how to evaluate a trained model
#         # checkpoint_path = "runs/rl_snake_multicircle_bc_30/models/snake_model_2500.pth"
#         # eval_metrics, final_snake_image = rl_snake.evaluate(
#         #     checkpoint_path=checkpoint_path,
#         #     setting_fn=setting_fn,  # Can use a different setting function for evaluation
#         #     num_episodes=1,  # Default is now 1
#         #     num_steps=200,
#         #     log_to_wandb=True
#         # )
#         # print("Evaluation Results:", eval_metrics)
#         # final_snake_image.show()  # Display the final snake image
        
#         # Example of comprehensive evaluation on all settings
#         # checkpoint_path = "runs/rl_snake_triangle_bc_30/models/snake_model_2500.pth"
#         # results = rl_snake.evaluate_on_all_settings(
#         #     checkpoint_path=checkpoint_path,
#         #     output_dir="output/triangle_model_eval",
#         #     num_steps=200,
#         #     log_to_wandb=True
#         # )
#         # print("Comprehensive evaluation completed!")
#         # print("Results:", results)
#     else:
#         img = circle_setting_fn()[0]

#         snake = Snake(initial_points=initial_snake, 
#                       external_energy=external_energy_fn(img),
#                       alpha=1.0, 
#                       beta=0.0,
#                       gamma=1.0)
#         snake_evolution = snake.optimize(num_iterations=1000)
#         visualize_snake_evolution(img, snake_evolution, title="Active Contour Evolution")
#     exit(0)

    img_height, img_width = 200, 200
    num_snake_points = 150

    RL = True

    def rect_setting_fn():
        img = create_centered_rectangular_mask(
            img_height,
            img_width,
            rect_height=img_height // 4,
            rect_width=img_width // 4,
        )
        return img, None

    def circle_setting_fn():
        img = create_circular_mask(img_height, img_width, radius=img_width // 3)
        return img, None

    def setting_fn():
        img = create_multi_circle_mask(
            img_height, img_width, num_circles=5, radius_range=(10, 30)
        )
        return img, None

    def triangle_setting_fn():
        img = create_triangle_mask(
            img_height,
            img_width,
            center_x=img_width // 2,
            center_y=img_height // 3,
            base=img_width // 3,
            triangle_height=img_height // 3,
            orientation="up",
        )
        return img, None

    def star_setting_fn():
        img = create_star_mask(
            img_height,
            img_width,
            center_x=img_width // 2,
            center_y=img_height // 3,
            outer_radius=img_width // 4,
            inner_radius=img_width // 8,
            num_points=5,
        )
        return img, None

    def ellipse_setting_fn():
        img = create_elliptical_mask(
            img_height,
            img_width,
            center_x=img_width // 2,
            center_y=img_height // 3,
            radius_x=img_width // 3,
            radius_y=img_height // 4,
            rotation_angle_rad=0,
        )
        return img, None

    def external_energy_fn(img):
        return GradientExternalEnergy(img)

    initial_snake = create_circle_points(
        num_snake_points,
        center_x=img_width // 2,
        center_y=img_height // 2,
        radius=img_width // 3,
    )

    if RL:
        shapes = {
            "star": star_setting_fn,
            "ellipse": ellipse_setting_fn,
            "rectangle": rect_setting_fn,
            "multi_circle": setting_fn,
            "circle": circle_setting_fn,
            "triangle": triangle_setting_fn,
        }

        for shape_name, shape_fn in shapes.items():
            # Create a unique run_name for each shape
            current_run_name = f"mlp_bc_shape_{shape_name}"
            rl_snake = RLSnake(
                initial_points=initial_snake,
                external_energy_fn=external_energy_fn,
                setting_fn=shape_fn,
                run_name="rl_snake_overfit1_bc",
                update_freq=1,
                save_freq=100,
                alpha=5e-5,
                beta=1e-6,
                gamma=0.2,
                model_type="mlp",
                obs_type="com,roi",
            )

            rl_snake.optimize(num_settings=3000)
            wandb.finish()
    else:
        img = circle_setting_fn()[0]

        snake = Snake(
            initial_points=initial_snake,
            external_energy=external_energy_fn(img),
            alpha=1.0,
            beta=0.0,
            gamma=1.0,
        )
        snake_evolution = snake.optimize(num_iterations=1000)
        visualize_snake_evolution(
            img, snake_evolution, title="Active Contour Evolution"
        )
    exit(0)


# if __name__ == "__main__":
#     generate_dataset(num_settings=1000, num_snake_points=150)
#     # img_height, img_width = 200, 200
#     # img = create_multi_circle_mask(img_height, img_width, num_circles=5, radius_range=(10, 25))
#     # # img = np.array(Image.open("test_images/eagle.jpeg").convert('L'))
#     # # img_height, img_width = img.shape

# num_snake_points = 150
# initial_snake = create_circle_points(num_snake_points,
#                                      center_x=img_width//2,
#                                      center_y=img_height//2,
#                                      radius=img_width//3)
# # initial_snake = create_ellipse_points(num_points=num_snake_points,
# #                                       center_x=img_width // 2,
# #                                       center_y=img_height // 2,
# #                                       radius_x=img_width // 3,
# #                                       radius_y=img_height // 3,
# #                                       rotation_angle_rad=0) # Example: 30 degrees rotation

# alpha_val = 0.05
# beta_val = 0.1

# gamma_A_matrix = 1.0

# num_iterations = 10000

# snake = Snake(initial_points=initial_snake,
#               external_energy=GradientExternalEnergy(img),
#               alpha=alpha_val,
#               beta=beta_val,
#               gamma=gamma_A_matrix)

# # visualize_gradient_fields(img, snake.ext_energy)

# snake_evolution = snake.optimize(num_iterations)


# visualize_snake_evolution(img, snake_evolution, title="Active Contour Evolution")
