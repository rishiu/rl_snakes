from code.utils import create_circle_points
from code.gradient_energy import GradientExternalEnergy
from code.snake import Snake
from code.images import get_image_setting_functions
from code.visualization import (
    plot_snake_on_image,
    visualize_gradient_fields,
    visualize_snake_evolution,
)
from code.rl_snake import RLSnake

import os
import wandb


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

    classical_shapes_fn_dict = get_image_setting_functions(
        img_height, img_width, exclude=["triangle", "star", "ellipse"]
    )

    for shape, fn in classical_shapes_fn_dict.items():
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

    rl_snake_params = {
        "num_snake_points": 150,
        "alpha": 5e-7,
        "beta": 1e-8,
        "gamma": 1.0,
        "num_settings": 1000,
        "model_types": ["mlp", "cnn"],
        "obs_types": ["com", "roi"],
        "trainer_algorithms": ["ppo", "bc"],
        "output_dir": os.path.join(images_dir, "rl_snake"),
    }
    rl_shapes_fn_dict = get_image_setting_functions(img_height, img_width)
    for trainer in rl_snake_params["trainer_algorithms"]:
        for shape, shape_fn in classical_shapes_fn_dict.items():
            img = shape_fn()[0]
            # For real images
            # img_height, img_width = img.shape

            def external_energy_fn(img):
                return GradientExternalEnergy(img)

            current_run_name = (
                f"{rl_snake_params["model_types"][0]}_{trainer}_shape_{shape}"
            )
            initial_snake = create_circle_points(
                rl_snake_params["num_snake_points"],
                center_x=img_width // 2,
                center_y=img_height // 2,
                radius=img_width // 3,
            )
            rl_snake = RLSnake(
                initial_points=initial_snake,
                external_energy_fn=external_energy_fn,
                setting_fn=shape_fn,
                # run_name=
                run_name=None,
                update_freq=1,
                save_freq=500,
                alpha=rl_snake_params["alpha"],
                beta=rl_snake_params["beta"],
                gamma=rl_snake_params["gamma"],
                model_type=rl_snake_params["model_types"][0],
                obs_type=rl_snake_params["obs_types"].join(","),
                trainer_algorithm=trainer,
            )
            rl_snake.optimize(num_settings=rl_snake_params["num_settings"])

            rl_snake.evaluate_on_all_settings(
                checkpoint_path="runs/rl_snake_multicircle_ppo_30/models/snake_model_1500.pth",
                num_steps=200,
            )
            wandb.finish()


# Example of how to evaluate a trained model
# checkpoint_path = "runs/rl_snake_multicircle_bc_30/models/snake_model_2500.pth"
# eval_metrics, final_snake_image = rl_snake.evaluate(
#     checkpoint_path=checkpoint_path,
#     setting_fn=setting_fn,  # Can use a different setting function for evaluation
#     num_episodes=1,  # Default is now 1
#     num_steps=200,
#     log_to_wandb=True
# )
# print("Evaluation Results:", eval_metrics)
# final_snake_image.show()  # Display the final snake image

# Example of comprehensive evaluation on all settings
# checkpoint_path = "runs/rl_snake_triangle_bc_30/models/snake_model_2500.pth"
# results = rl_snake.evaluate_on_all_settings(
#     checkpoint_path=checkpoint_path,
#     output_dir="output/triangle_model_eval",
#     num_steps=200,
#     log_to_wandb=True
# )
# print("Comprehensive evaluation completed!")
# print("Results:", results)
