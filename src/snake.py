import numpy as np
from tqdm import tqdm
from src.utils import ma


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
