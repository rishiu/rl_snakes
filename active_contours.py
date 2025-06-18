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
        update_freq=5,
        save_freq=20,
        alpha=0.01,
        beta=0.01,
        gamma=0.01,
        model_type="mlp",
        image_channels=1,
        obs_type="trad",
        device="cpu",
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
        self.device = device

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

        # self.vector_input_dim = self.num_control_points * 2 + (
        #     32 * 32
        # )  # self.get_obs(self.obs_type).shape[0]

        calculated_vector_dim = 0
        temp_obs_types = self.obs_type.split(",")

        if "trad" in temp_obs_types:
            calculated_vector_dim += (
                self.num_control_points * 2 + self.num_control_points
            )  # points (x,y) + image_vals
        if "com" in temp_obs_types:
            calculated_vector_dim += (
                self.num_control_points * 2
            )  # relative_vectors.flatten()

        if self.model_type == "mlp" and "roi" in temp_obs_types:
            # For MLP, if 'roi' is included, its flattened size is part of the vector_input_dim
            # Assumes target_size for ROI is (200, 200) and 1 channel
            calculated_vector_dim += 200 * 200 * self.image_channels

        self.vector_input_dim = calculated_vector_dim

        self.output_dim = self.num_control_points * 2

        # self.trainer = PPOTrainer(self.vector_input_dim, self.output_dim, \
        #                           actor_lr=5e-5, critic_lr=5e-5, \
        #                           gamma=0.99, K_epochs=self.update_freq, eps_clip=0.2, \
        #                           action_std_init=0.1,
        #                           model_type=self.model_type,
        #                           image_channels=self.image_channels)

        self.trainer = BCTrainer(
            self.vector_input_dim,
            self.output_dim,
            actor_lr=1e-4,
            critic_lr=5e-5,
            gamma=0.99,
            K_epochs=self.update_freq,
            eps_clip=0.2,
            action_std_init=0.1,
            model_type=self.model_type,
            image_channels=self.image_channels,
        )

        A = Snake.setup_A(self.num_control_points, 0.05, 0.1)
        A_ = A + 1.0 * np.eye(self.num_control_points)

        def gt_action_fn(obs):
            if self.model_type == "cnn":
                vector_tensor_obs = obs[1]

                snake_points_vector = vector_tensor_obs.squeeze(0)[
                    : self.num_control_points * 2
                ]
                snake_points = np.array(
                    snake_points_vector.reshape(2, -1).cpu()
                )  # Convert to numpy and reshape

            else:
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

        self.trainer.set_gt_action_fn(gt_action_fn)

        self.trainer.set_gt_action_fn(gt_action_fn)

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
        vector_features_list = []

        if "trad" in obs_types:
            vector_features_list.append(self.get_trad_obs())
        if "com" in obs_types:
            vector_features_list.append(self.get_com_obs())

        if vector_features_list:
            vector_input_tensor = torch.cat(vector_features_list, dim=0).float()
        else:
            vector_input_tensor = torch.empty(0).float()
        vector_input_tensor = vector_input_tensor.unsqueeze(0).to(self.device)

        image_input_tensor = None
        if "roi" in obs_types:
            image_input_array = self.get_image_roi(
                self.current_points, target_size=(200, 200)
            )
            image_input_tensor = torch.tensor(image_input_array, dtype=torch.float32)

            if image_input_tensor.ndim == 2:
                image_input_tensor = image_input_tensor.unsqueeze(0)
            image_input_tensor = image_input_tensor.unsqueeze(0)
            image_input_tensor = image_input_tensor.to(self.device)

        if self.model_type == "mlp":
            if "roi" in obs_types:
                flattened_roi = image_input_tensor.flatten(start_dim=1)
                return torch.cat((vector_input_tensor, flattened_roi), dim=1).squeeze(0)
            else:
                return vector_input_tensor.squeeze(0)
        elif self.model_type == "cnn":
            if image_input_tensor is None and "roi" in obs_types:
                raise ValueError(
                    "CNN model expected 'roi' for image_input, but it was not generated correctly."
                )
            return (image_input_tensor, vector_input_tensor)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type} in get_obs")

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
        return vector_features

    # This helper function (get_image_roi) is called by get_obs and must be correctly defined
    def get_image_roi(self, points, target_size=(200, 200)):
        min_x = int(np.min(points[0, :]))
        max_x = int(np.max(points[0, :]))
        min_y = int(np.min(points[1, :]))
        max_y = int(np.max(points[1, :]))

        img_height, img_width = self.curr_img.shape[:2]
        min_x = np.clip(min_x, 0, img_width)
        max_x = np.clip(max_x, 0, img_width)
        min_y = np.clip(min_y, 0, img_height)
        max_y = np.clip(max_y, 0, img_height)

        if min_x >= max_x or min_y >= max_y:
            roi = np.zeros(target_size, dtype=self.curr_img.dtype)
        else:
            roi = self.curr_img[min_y:max_y, min_x:max_x]

        roi_normalized = (roi / 255.0 - 0.5) * 2.0
        roi_resized = cv2.resize(
            roi_normalized,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return roi_resized

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


if __name__ == "__main__":
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
        img = create_circular_mask(img_height, img_width, radius=img_width // 4)
        return img, None

    def setting_fn():
        img = create_multi_circle_mask(
            img_height, img_width, num_circles=5, radius_range=(10, 25)
        )
        return img, None

    # dataset_setting_fn = build_dataset_setting_fn("./datasets/")

    def external_energy_fn(img):
        return GradientExternalEnergy(img)

    initial_snake = create_circle_points(
        num_snake_points,
        center_x=img_width // 2,
        center_y=img_height // 2,
        radius=img_width // 3,
    )

    if RL:
        rl_snake = RLSnake(
            initial_points=initial_snake,
            external_energy_fn=external_energy_fn,
            setting_fn=circle_setting_fn,
            run_name="rl_snake_overfit1_bc",
            update_freq=1,
            save_freq=100,
            alpha=5e-7,
            beta=1e-8,
            # options: "mlp", "cnn"
            model_type="cnn",
            obs_type="com,roi",
        )

        rl_snake.optimize(num_settings=3000)
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
