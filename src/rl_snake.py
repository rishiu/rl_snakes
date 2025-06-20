import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from src.gradient_energy import GradientExternalEnergy

from src.utils import (
    bilinear_interpolate,
    calculate_normals,
)
from src.visualization import (
    plot_snake_on_image,
    visualize_snake_evolution_with_energy_rewards,
)
from src.mask_utils import (
    create_multi_circle_mask,
    create_rectangular_mask,
    create_circular_mask,
    create_triangle_mask,
    create_star_mask,
    create_elliptical_mask,
)

from src.snake import Snake
from src.rl_train_algorithm.ppo_trainer import PPOTrainer
from src.rl_train_algorithm.bc_trainer import BCTrainer
from src.images import get_image_setting_functions
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
        output_dir,
        update_freq=1,
        save_freq=20,
        alpha=0.01,
        beta=0.01,
        gamma=0.01,
        model_type="mlp",
        image_channels=1,
        obs_type="trad",
        trainer_algorithm="bc",
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
        self.trainer_algorithm = trainer_algorithm
        self.output_dir = output_dir

        os.makedirs(f"runs/{run_name}/models", exist_ok=True)
        self.run = wandb.init(
            project="RLSnake",
            name=self.run_name,
            config={
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "update_freq": self.update_freq,
                "save_freq": self.save_freq,
                "run_name": self.run_name,
                "num_control_points": self.num_control_points,
                "model_type": self.model_type,
            },
        )

        self.vector_input_dim = self.num_control_points * 2 + (
            32 * 32
        )  # self.get_obs(self.obs_type).shape[0]
        self.output_dim = self.num_control_points * 2

        if self.trainer_algorithm == "ppo":
            self.trainer = PPOTrainer(
                self.vector_input_dim,
                self.output_dim,
                actor_lr=5e-5,
                critic_lr=5e-5,
                gamma=0.99,
                K_epochs=self.update_freq,
                eps_clip=0.2,
                action_std_init=0.1,
                model_type=self.model_type,
                image_channels=self.image_channels,
            )
        elif self.trainer_algorithm == "bc":
            self.trainer = BCTrainer(
                self.vector_input_dim,
                self.output_dim,
                actor_lr=5e-5,
                critic_lr=5e-5,
                gamma=0.99,
                K_epochs=self.update_freq,
                eps_clip=0.2,
                action_std_init=0.1,
                model_type=self.model_type,
                image_channels=self.image_channels,
            )

            def gt_action_fn(obs):
                snake_points = np.array(
                    obs[0, : self.num_control_points * 2].reshape(2, -1)
                )

                f_x, f_y = self.ext_energy(snake_points)
                dext = np.array([f_x, f_y])

                y = 1.0 * snake_points - dext

                new_snake_points = np.zeros_like(snake_points)
                A = Snake.setup_A(self.num_control_points, 0.05, 0.1)
                A_ = A + 1.0 * np.eye(self.num_control_points)
                new_snake_points[0, :] = np.linalg.solve(A_, y[0, :])
                new_snake_points[1, :] = np.linalg.solve(A_, y[1, :])
                return new_snake_points - snake_points

            self.trainer.set_gt_action_fn(gt_action_fn)
        else:
            raise ValueError(f"Unsupported trainer_algorithm: {self.trainer_algorithm}")

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

        self.curr_img = img

        def external_energy_fn(img):
            return GradientExternalEnergy(img)

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

        self.current_points = self.initial_points.copy()

        return rewards, reward_by_time, snake_fig

    def optimize(self, num_settings=100, num_steps=200):
        avg_rewards_by_time = np.zeros(num_steps)
        for i in tqdm(range(num_settings)):
            rewards, reward_by_time, snake_fig = self.run_setting(
                self.setting_fn, i, num_steps
            )
            wandb.log({"reward": np.mean(rewards), "final_reward": rewards[-1]}, step=i)
            avg_rewards_by_time += reward_by_time

            if i % self.update_freq == 0:
                bc_loss = self.trainer.update(self.last_value)
                wandb.log({"bc_loss": bc_loss}, step=i)
            if i % self.save_freq == 0:
                snake_fig.savefig(
                    os.path.join(self.output_dir, f"training_iter_{i}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(snake_fig)
                self.trainer.save(f"runs/{self.run_name}/models/rl_snake_model_{i}.pth")
                avg_rewards_by_time = np.zeros(num_steps)
            else:
                plt.close(snake_fig)
        wandb.finish()

    def evaluate(
        self,
        checkpoint_path,
        setting_fn=None,
        num_episodes=1,
        num_steps=200,
        log_to_wandb=False,
        shape_name="",
    ):
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

            def external_energy_fn(img):
                return GradientExternalEnergy(img)

            self.ext_energy = external_energy_fn(img)

            episode_reward = 0
            step_count = 0
            internal_energy_episode = []
            external_energy_episode = []

            if log_to_wandb:
                wandb.log({f"{shape_name}_eval_img": wandb.Image(img)}, step=episode)

            # Run episode
            for step in range(num_steps):
                obs = self.get_obs(self.obs_type)
                offset = self.trainer.select_action(obs).reshape(
                    2, self.num_control_points
                )

                self.current_points = self.current_points + offset
                self.current_points[0, :] = np.clip(
                    self.current_points[0, :], 0, self.curr_img.shape[0] - 1
                )
                self.current_points[1, :] = np.clip(
                    self.current_points[1, :], 0, self.curr_img.shape[1] - 1
                )

                internal_energy = self.compute_internal_energy()
                external_energy = self.compute_external_energy()
                internal_energy_episode.append(internal_energy)
                external_energy_episode.append(external_energy)

                # Calculate reward (same logic as in run_setting)
                if gt_points is not None:
                    reward = (
                        np.exp(-0.1 * np.mean(np.abs(self.current_points - gt_points)))
                        + external_energy
                        - internal_energy
                    )
                else:
                    reward = 0.1 * (external_energy + np.exp(-0.1 * internal_energy))

                episode_reward += reward
                step_count += 1

                # Check termination conditions
                if internal_energy > 0.1:
                    break
                elif np.abs(external_energy - internal_energy) < 1e-5:
                    break
                elif (
                    gt_points is not None
                    and np.mean(np.abs(self.current_points - gt_points)) < 0.1
                ):
                    break

            episode_rewards.append(episode_reward)
            final_rewards.append(reward)
            episode_lengths.append(step_count)
            internal_energies.extend(internal_energy_episode)
            external_energies.extend(external_energy_episode)

            # Generate final snake image (for the last episode or if only 1 episode)
            if episode == num_episodes - 1:  # Save the final episode's image
                final_snake_image = plot_snake_on_image(
                    self.curr_img, self.current_points
                )
                if log_to_wandb:
                    wandb.log(
                        {f"{shape_name}_eval_snake": wandb.Image(final_snake_image)},
                        step=episode,
                    )

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
            "episode_lengths": episode_lengths,
        }

        # Log summary metrics
        if log_to_wandb:
            wandb.log(
                {
                    f"{shape_name}_eval_mean_episode_reward": eval_metrics[
                        "mean_episode_reward"
                    ],
                    f"{shape_name}_eval_std_episode_reward": eval_metrics[
                        "std_episode_reward"
                    ],
                    f"{shape_name}_eval_mean_final_reward": eval_metrics[
                        "mean_final_reward"
                    ],
                    f"{shape_name}_eval_mean_episode_length": eval_metrics[
                        "mean_episode_length"
                    ],
                    f"{shape_name}_eval_mean_internal_energy": eval_metrics[
                        "mean_internal_energy"
                    ],
                    f"{shape_name}_eval_mean_external_energy": eval_metrics[
                        "mean_external_energy"
                    ],
                }
            )

        print(f"Evaluation completed!")
        print(
            f"Mean episode reward: {eval_metrics['mean_episode_reward']:.4f} ± {eval_metrics['std_episode_reward']:.4f}"
        )
        print(
            f"Mean final reward: {eval_metrics['mean_final_reward']:.4f} ± {eval_metrics['std_final_reward']:.4f}"
        )
        print(
            f"Mean episode length: {eval_metrics['mean_episode_length']:.2f} ± {eval_metrics['std_episode_length']:.2f}"
        )

        return eval_metrics, final_snake_image

    def evaluate_on_all_settings(
        self,
        checkpoint_path,
        output_dir="output",
        num_steps=200,
        log_to_wandb=False,
    ):
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

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define all setting functions
        img_height, img_width = 200, 200

        rl_shapes_fn_dict = get_image_setting_functions(img_height, img_width)

        # Store results
        all_results = {}

        # Load the model once
        print(f"Loading model from checkpoint: {checkpoint_path}")
        self.trainer.load(checkpoint_path)
        wandb.init(
            project="RLSnake",
            name=f"{self.run_name}_eval",
        )

        # Evaluate on each setting
        for setting_name, setting_fn in rl_shapes_fn_dict.items():
            try:
                # Run evaluation
                eval_metrics, final_snake_image = self.evaluate(
                    checkpoint_path=checkpoint_path,
                    setting_fn=setting_fn,
                    num_episodes=1,
                    num_steps=num_steps,
                    log_to_wandb=True,
                    shape_name=setting_name,
                )

                # Save the image
                image_path = os.path.join(output_dir, f"{setting_name}_evaluation.png")
                final_snake_image.savefig(image_path, dpi=150, bbox_inches="tight")
                plt.close(final_snake_image)  # Close to free memory

                # Store results
                all_results[setting_name] = {
                    "metrics": eval_metrics,
                    "image_path": image_path,
                }

                print(
                    f"✓ {setting_name}: Reward={eval_metrics['mean_episode_reward']:.4f}, "
                    f"Length={eval_metrics['mean_episode_length']:.1f} steps, "
                    f"Image saved to {image_path}"
                )

            except Exception as e:
                print(f"✗ Failed to evaluate {setting_name}: {str(e)}")
                all_results[setting_name] = {
                    "metrics": None,
                    "image_path": None,
                    "error": str(e),
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
                print(
                    f"{setting_name:<15} {reward:<10.4f} {length:<10.1f} {status:<15}"
                )
            else:
                print(f"{setting_name:<15} {'N/A':<10} {'N/A':<10} {'✗ Failed':<15}")

        # Log summary to wandb if enabled
        if log_to_wandb:
            summary_metrics = {}
            for setting_name, result in all_results.items():
                if result["metrics"] is not None:
                    summary_metrics[f"eval_all_{setting_name}_reward"] = result[
                        "metrics"
                    ]["mean_episode_reward"]
                    summary_metrics[f"eval_all_{setting_name}_length"] = result[
                        "metrics"
                    ]["mean_episode_length"]
            wandb.log(summary_metrics)
            wandb.finish()

        return all_results
