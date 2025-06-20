from src.mask_utils import (
    create_circular_mask,
    create_rectangular_mask,
    create_triangle_mask,
    create_star_mask,
    create_elliptical_mask,
    create_multi_circle_mask,
)


def get_image_setting_functions(img_height, img_width, exclude=[]):
    """
    Returns a dictionary of image setting functions for various shapes.
    Each function takes no arguments and returns (img, None).
    img_height and img_width are captured via closure.
    """

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

    # def test_eagle_image_fn():
    #     img = np.array(Image.open("test_images/eagle.jpeg").convert("L"))
    #     return img, None

    all_shapes = {
        "circle": circle_setting_fn,
        "multi_circle": multi_circle_setting_fn,
        "rectangle": rectangle_setting_fn,
        "triangle": triangle_setting_fn,
        "star": star_setting_fn,
        "ellipse": ellipse_setting_fn,
        # "test_eagle": test_eagle_image_fn,
    }

    return {name: fn for name, fn in all_shapes.items() if name not in exclude}
