from code.utils import create_circle_points
from code.gradient_energy import GradientExternalEnergy
from code.snake import Snake
from code.images import get_image_setting_functions
from code.visualization import (
    plot_snake_on_image,
    visualize_gradient_fields,
    visualize_snake_evolution,
)

import os


if __name__ == "__main__":
    img_height, img_width = 200, 200

    images_dir = "images"

    classical_snake_params = {
        "num_snake_points": 150,
        "alpha": 1.0,
        "beta": 0.05,
        "gamma": 1.0,
        "num_iterations": 1000,
        "output_dir": os.path.join(images_dir, "classical_snake"),
    }

    shapes_fn_dict = get_image_setting_functions(
        img_height, img_width, exclude=["triangle", "star", "ellipse"]
    )

    for shape, fn in shapes_fn_dict.items():
        shape_output_dir = os.path.join(classical_snake_params["output_dir"], shape)
        os.makedirs(shape_output_dir, exist_ok=True)
        initial_snake = create_circle_points(
            classical_snake_params["num_snake_points"],
            center_x=img_width // 2,
            center_y=img_height // 2,
            radius=img_width // 3,
        )
        img = fn()[0]
        snake = Snake(
            initial_points=initial_snake,
            external_energy=GradientExternalEnergy(img),
            alpha=classical_snake_params["alpha"],
            beta=classical_snake_params["beta"],
            gamma=classical_snake_params["gamma"],
        )
        snake_evolution = snake.optimize(
            num_iterations=classical_snake_params["num_iterations"]
        )
        # visualize_snake_evolution(img, snake_evolution, title="Active Contour Evolution")

        initial = plot_snake_on_image(
            img, snake_evolution[0], title="Classical Snake: Initial Position"
        )
        initial.savefig(
            os.path.join(shape_output_dir, "initial.png"),
            dpi=150,
            bbox_inches="tight",
        )
        final = plot_snake_on_image(
            img, snake_evolution[-1], title="Classical Snake: Final Position"
        )
        final.savefig(
            os.path.join(shape_output_dir, "final.png"),
            dpi=150,
            bbox_inches="tight",
        )
        gradients = visualize_gradient_fields(img, GradientExternalEnergy(img))
        gradients.savefig(
            os.path.join(shape_output_dir, "external_energy_gradient_map.png"),
            dpi=150,
            bbox_inches="tight",
        )
