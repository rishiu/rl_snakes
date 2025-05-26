import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ext_energy.gradient_energy import GradientExternalEnergy
from utils import ma, create_circle_points, create_ellipse_points, bilinear_interpolate
from viz import visualize_snake_evolution, visualize_gradient_fields, plot_snake_on_image
from mask_utils import create_multi_circle_mask
from RL.ppo_trainer import PPOTrainer
import os
import wandb
import matplotlib.pyplot as plt

class RLSnake:
    def __init__(self, 
                 initial_points, 
                 external_energy_fn, 
                 setting_fn, 
                 run_name,
                 update_freq=5,
                 save_freq=20,
                 alpha=0.01, 
                 beta=0.01, 
                 gamma=0.01,
                 ):
        self.num_control_points = initial_points.shape[1]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_points = initial_points
        self.initial_points = initial_points.copy()
        self.ext_energy = None
        self.curr_img = None
        self.setting_fn = setting_fn
        self.update_freq = update_freq
        self.save_freq = save_freq
        self.run_name = run_name

        os.makedirs(f"runs/{run_name}/models", exist_ok=True)
        self.run = wandb.init(project="RLSnake", name=run_name,
                              config={
                                  "alpha": alpha,
                                  "beta": beta,
                                  "gamma": gamma,
                                  "update_freq": update_freq,
                                  "save_freq": save_freq,
                                  "run_name": run_name,
                                  "num_control_points": self.num_control_points,
                              })

        self.input_dim = self.num_control_points * 3
        self.output_dim = self.num_control_points * 2

        self.trainer = PPOTrainer(self.input_dim, self.output_dim, \
                                  actor_lr=5e-4, critic_lr=5e-4, \
                                  gamma=0.99, K_epochs=self.update_freq, eps_clip=0.2, \
                                  action_std_init=0.1)
        

    def compute_internal_energy(self):
        diff = np.power(self.current_points - np.roll(self.current_points, 1, axis=1), 2)
        
        diff_ = np.power(diff - np.roll(diff, 1, axis=1), 2)
        return self.alpha * np.sum(diff) + self.beta * np.sum(diff_)

    def compute_external_energy(self):
        return self.ext_energy.get_energy(self.current_points)
    
    def sample_image(self):
        return bilinear_interpolate(self.curr_img, self.current_points)
    
    def get_obs(self):
        points = self.current_points.copy().flatten()
        points = torch.tensor(points, dtype=torch.float32)
        img_vals = torch.tensor(self.sample_image(), dtype=torch.float32)
        x = torch.cat([points, img_vals], dim=0).unsqueeze(0)
        return x
    
    def run_setting(self, setting_fn, setting_idx):
        img, gt_points = setting_fn()

        wandb.log({"img": wandb.Image(img)}, step=setting_idx)
        
        self.curr_img = img
        self.ext_energy = external_energy_fn(img)
        
        rewards = []

        for step in range(10):
            obs = self.get_obs()
            offset = self.trainer.select_action(obs).reshape(2, self.num_control_points)

            self.current_points = self.current_points + offset
            self.current_points[0,:] = np.clip(self.current_points[0,:], 0, self.curr_img.shape[0] - 1)
            self.current_points[1,:] = np.clip(self.current_points[1,:], 0, self.curr_img.shape[1] - 1)

            reward = -1e-3 * (self.compute_internal_energy() + self.compute_external_energy())

            rewards.append(reward)
            done = step == 1000

            self.trainer.add_reward_and_done(reward, done)

        self.last_value = self.trainer.get_value(self.get_obs())

        snake_fig = plot_snake_on_image(self.curr_img, self.current_points)
        wandb.log({"snake": snake_fig}, step=setting_idx)
        plt.close(snake_fig)

        self.current_points = self.initial_points.copy()

        return rewards

    def optimize(self, num_settings=100):
        for i in tqdm(range(num_settings)):
            rewards = self.run_setting(self.setting_fn, i)
            wandb.log({"reward": np.mean(rewards), "final_reward": rewards[-1]}, step=i)

            if i % self.update_freq == 0:
                self.trainer.update(self.last_value)
            if i % self.save_freq == 0:
                self.trainer.save(f"runs/{self.run_name}/models/snake_model_{i}.pth")

class Snake:
    def __init__(self, initial_points, external_energy, alpha=0.01, beta=0.01, gamma=0.01):
        if not isinstance(initial_points, np.ndarray) or initial_points.ndim != 2 or initial_points.shape[0] != 2:
            raise ValueError("initial_points must be a NumPy array of shape (2, N)")
        
        self.num_control_points = initial_points.shape[1]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_points = initial_points
        self.ext_energy = external_energy

    def setup_A(self, alphas, betas):
        N = self.num_control_points
        if np.isscalar(alphas):
            alphas = np.full(N, alphas)
        if np.isscalar(betas):
            betas = np.full(N, betas)
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = ma(alphas, i) + ma(alphas, i + 1) + 4 * ma(betas, i) + ma(betas, i + 1) + ma(betas, i - 1)
            A[i, (i-1) % N] = -ma(alphas, i) - 2 * ma(betas, i) - 2 * ma(betas, i - 1)
            A[i, (i+1) % N] = -ma(alphas, i+1) - 2 * ma(betas, i) - 2 * ma(betas, i + 1)
            A[i, (i-2) % N] = ma(betas, i-1)
            A[i, (i+2) % N] = ma(betas, i+1)
        return A
    
    def optimize(self, num_iterations=100):
        A = self.setup_A(self.alpha, self.beta)

        A_ = A + self.gamma * np.eye(self.num_control_points)

        snake_points = self.current_points.copy()

        snake_evolution = []

        for i in tqdm(range(num_iterations)):
            snake_evolution.append(snake_points.copy())

            f_x, f_y = self.ext_energy(snake_points)
            dext = np.array([f_x, f_y])

            y = self.gamma * snake_points - dext

            # print(snake_points)
            snake_points[0,:] = np.linalg.solve(A_, y[0,:])#solve_banded((2, 2), ab, y[0,:])
            snake_points[1,:] = np.linalg.solve(A_, y[1,:])#solve_banded((2, 2), ab, y[1,:])

            # print(snake_points)

            # print("----------\n")

        snake_evolution.append(snake_points.copy())
        # print(snake_evolution)
        self.current_points = snake_points.copy()
        return snake_evolution


# if __name__ == "__main__":
#     img_height, img_width = 200, 200
#     num_snake_points = 150

#     def setting_fn():
#         img = create_multi_circle_mask(img_height, img_width, num_circles=5, radius_range=(10, 25))
#         return img
    
#     def external_energy_fn(img):
#         return GradientExternalEnergy(img)
    
#     initial_snake = create_circle_points(num_snake_points, 
#                                          center_x=img_width//2, 
#                                          center_y=img_height//2, 
#                                          radius=img_width//3)
    
#     rl_snake = RLSnake(initial_points=initial_snake, 
#                        external_energy_fn=external_energy_fn,
#                        setting_fn=setting_fn,
#                        run_name="rl_snake_200",
#                        update_freq=5,
#                        save_freq=20,
#                        alpha=5e-3,
#                        beta=1e-3)

#     rl_snake.optimize(num_settings=1000)

#     exit(0)

def generate_dataset(num_settings=1000, num_snake_points=150):
    os.makedirs("datasets", exist_ok=True)

    img_height, img_width = 200, 200

    alpha_val = 0.05
    beta_val = 0.1
    gamma_A_matrix = 1.0

    num_iterations = 10000

    initial_snake = create_circle_points(num_snake_points, 
                                         center_x=img_width//2, 
                                         center_y=img_height//2, 
                                         radius=img_width//3)

    for i in tqdm(range(num_settings)):
        os.makedirs(f"datasets/snake_evolution_{i}", exist_ok=True)

        img = create_multi_circle_mask(img_height, img_width, num_circles=5, radius_range=(10, 25))
        external_energy = GradientExternalEnergy(img)
        snake = Snake(initial_points=initial_snake, 
                      external_energy=external_energy,
                      alpha=alpha_val, 
                      beta=beta_val, 
                      gamma=gamma_A_matrix)
        snake_evolution = snake.optimize(num_iterations)

        I = Image.fromarray(img)
        I.save(f"datasets/snake_evolution_{i}/img.png")

        np.save(f"datasets/snake_evolution_{i}/snake_evolution.npy", snake_evolution)

        fig = plot_snake_on_image(img, snake_evolution[-1])
        fig.savefig(f"datasets/snake_evolution_{i}/snake_evolution.png")
        plt.close(fig)

if __name__ == "__main__":
    generate_dataset(num_settings=1000, num_snake_points=150)
    # img_height, img_width = 200, 200
    # img = create_multi_circle_mask(img_height, img_width, num_circles=5, radius_range=(10, 25))
    # # img = np.array(Image.open("test_images/eagle.jpeg").convert('L'))
    # # img_height, img_width = img.shape

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
