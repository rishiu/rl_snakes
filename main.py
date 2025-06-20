from code.mask_utils import (
    create_circular_mask,
    create_rectangular_mask,
    create_triangle_mask,
    create_star_mask,
    create_elliptical_mask,
    create_multi_circle_mask,
)
from code.utils import create_circle_points
from code.gradient_energy import GradientExternalEnergy
from code.snake import Snake
from code.visualization import (
    plot_snake_on_image,
    visualize_gradient_fields,
    visualize_snake_evolution,
)

import os

img_height, img_width = 200, 200


def circle_setting_fn():
    img = create_circular_mask(img_height, img_width, radius=img_width // 3)
    return img, None


def rectangle_setting_fn():
    img = create_rectangular_mask(
        img_height,
        img_width,
        rect_height=img_height // 4,
        rect_width=img_width // 4,
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


def multi_circle_setting_fn():
    img = create_multi_circle_mask(
        img_height, img_width, num_circles=5, radius_range=(10, 30)
    )
    return img, None


if __name__ == "__main__":
    images_dir = "images"

    classical_snake_params = {
        "num_snake_points": 150,
        "alpha": 1.0,
        "beta": 0.05,
        "gamma": 1.0,
        "num_iterations": 1000,
        "output_dir": os.path.join(images_dir, "classical_snake"),
    }

    shapes_to_test = {
        "circle": circle_setting_fn,
        "multi_circle": multi_circle_setting_fn,
        "rectangle": rectangle_setting_fn,
        # "triangle": triangle_setting_fn,
        # "star": star_setting_fn,
        # "ellipse": ellipse_setting_fn,
    }

    for shape, fn in shapes_to_test.items():
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
