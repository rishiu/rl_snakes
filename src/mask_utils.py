import numpy as np
import random
import cv2
import math


def create_circular_mask(height, width, center_x=None, center_y=None, radius=None):
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    if radius is None:
        radius = min(height, width) // 4

    Y, X = np.ogrid[:height, :width]
    dist_from_center_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    mask = dist_from_center_sq <= radius**2
    return mask.astype(np.uint8) * 255


def create_rectangular_mask(height, width, rect_height=None, rect_width=None):
    """
    Creates a mask with a centered rectangle.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.
        rect_height (int, optional): Height of the rectangle.
                                     Defaults to height // 2.
        rect_width (int, optional): Width of the rectangle.
                                    Defaults to width // 2.

    Returns:
        np.ndarray: The generated mask with a centered rectangle (dtype=np.uint8),
                    where the rectangle is 255 and the background is 0.
    """
    if rect_height is None:
        rect_height = height // 2
    if rect_width is None:
        rect_width = width // 2

    # Initialize the mask with zeros
    mask = np.zeros((height, width), dtype=np.uint8)

    # Calculate the top-left corner of the rectangle
    # Integer division ensures that if the rectangle cannot be perfectly centered,
    # it is shifted towards the top-left.
    start_x = (width - rect_width) // 2
    start_y = (height - rect_height) // 2

    # Calculate the bottom-right corner (exclusive for slicing)
    end_x = start_x + rect_width
    end_y = start_y + rect_height

    # Ensure the rectangle coordinates are within the mask boundaries
    # This handles cases where rect_height/rect_width might be larger than
    # the mask dimensions, or negative (though they should be positive).
    actual_start_y = max(0, start_y)
    actual_end_y = min(height, end_y)
    actual_start_x = max(0, start_x)
    actual_end_x = min(width, end_x)

    # Set the rectangle area to 255 if the slice is valid
    if actual_start_x < actual_end_x and actual_start_y < actual_end_y:
        mask[actual_start_y:actual_end_y, actual_start_x:actual_end_x] = 255

    return mask


def create_multi_circle_mask(height, width, num_circles, radius_range):
    """Generates a mask with multiple randomly placed circles.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.
        num_circles (int): Number of circles to generate.
        radius_range (tuple): (min_radius, max_radius) for the circles.

    Returns:
        np.ndarray: The generated mask with multiple circles.
    """
    final_mask = np.zeros((height, width), dtype=np.uint8)
    min_r, max_r = radius_range

    img_center_x = width // 2
    img_center_y = height // 2
    center_constraint_pixels = 20

    for _ in range(num_circles):
        radius = random.randint(min_r, max_r)

        # Determine bounds for center_x
        # Constraint from image center (must be within X pixels of image center)
        cx_min_constrained_by_center = img_center_x - center_constraint_pixels
        cx_max_constrained_by_center = img_center_x + center_constraint_pixels

        # Constraint from image boundaries (circle must be mostly within image)
        # random.randint's upper bound is inclusive.
        # So, center_x_max = width - radius - 1 means max x-coord of circle is (width - radius - 1) + radius = width - 1.
        cx_min_constrained_by_boundary = radius
        cx_max_constrained_by_boundary = width - radius - 1

        # Combined constraints for center_x
        actual_cx_min = max(
            cx_min_constrained_by_boundary, cx_min_constrained_by_center
        )
        actual_cx_max = min(
            cx_max_constrained_by_boundary, cx_max_constrained_by_center
        )

        # Determine bounds for center_y
        # Constraint from image center
        cy_min_constrained_by_center = img_center_y - center_constraint_pixels
        cy_max_constrained_by_center = img_center_y + center_constraint_pixels

        # Constraint from image boundaries
        cy_min_constrained_by_boundary = radius
        cy_max_constrained_by_boundary = height - radius - 1

        # Combined constraints for center_y
        actual_cy_min = max(
            cy_min_constrained_by_boundary, cy_min_constrained_by_center
        )
        actual_cy_max = min(
            cy_max_constrained_by_boundary, cy_max_constrained_by_center
        )

        # If the constraints make it impossible to place a circle,
        # random.randint will raise a ValueError if actual_min > actual_max.
        # This behavior is consistent with the original code if radius is too large for dimensions.
        if actual_cx_min > actual_cx_max or actual_cy_min > actual_cy_max:
            # Skip this circle if no valid center can be found.
            # Alternatively, could raise an error or try to adjust parameters.
            # For this problem, skipping seems like a reasonable default.
            # print(f"Warning: Cannot place circle with radius {radius} under given constraints. Skipping.")
            continue

        center_x = random.randint(actual_cx_min, actual_cx_max)
        center_y = random.randint(actual_cy_min, actual_cy_max)

        Y, X = np.ogrid[:height, :width]
        dist_from_center_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
        circle_mask = dist_from_center_sq <= radius**2
        final_mask = np.logical_or(final_mask, circle_mask)

    return final_mask.astype(np.uint8) * 255


def create_triangle_mask(
    height,
    width,
    center_x=None,
    center_y=None,
    base=None,
    triangle_height=None,
    orientation="up",
):
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    if base is None:
        base = width // 4
    if triangle_height is None:
        triangle_height = height // 4

    mask = np.zeros((height, width), dtype=np.uint8)

    half_base = base // 2

    if orientation == "up":
        p1 = (center_x - half_base, center_y + triangle_height // 2)
        p2 = (center_x + half_base, center_y + triangle_height // 2)
        p3 = (center_x, center_y - triangle_height // 2)
    else:  # 'down'
        p1 = (center_x - half_base, center_y - triangle_height // 2)
        p2 = (center_x + half_base, center_y - triangle_height // 2)
        p3 = (center_x, center_y + triangle_height // 2)

    vertices = np.array([p1, p2, p3], dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def create_star_mask(
    height,
    width,
    center_x=None,
    center_y=None,
    outer_radius=None,
    inner_radius=None,
    num_points=5,
):
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    if outer_radius is None:
        outer_radius = min(height, width) // 4
    if inner_radius is None:
        inner_radius = outer_radius // 2
    if num_points < 2:
        raise ValueError("num_points must be at least 2 for a star shape.")

    mask = np.zeros((height, width), dtype=np.uint8)
    points = []

    for i in range(2 * num_points):
        angle = math.pi * i / num_points
        r = outer_radius if i % 2 == 0 else inner_radius
        x = center_x + r * math.cos(angle - math.pi / 2)
        y = center_y + r * math.sin(angle - math.pi / 2)
        points.append([int(x), int(y)])

    vertices = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def create_elliptical_mask(
    height,
    width,
    center_x=None,
    center_y=None,
    radius_x=None,
    radius_y=None,
    rotation_angle_rad=0,
):
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    if radius_x is None:
        radius_x = min(height, width) // 4
    if radius_y is None:
        radius_y = min(height, width) // 4

    Y, X = np.ogrid[:height, :width]

    X_shifted = X - center_x
    Y_shifted = Y - center_y

    cos_phi = np.cos(rotation_angle_rad)
    sin_phi = np.sin(rotation_angle_rad)

    X_rotated = X_shifted * cos_phi + Y_shifted * sin_phi
    Y_rotated = -X_shifted * sin_phi + Y_shifted * cos_phi

    mask = (X_rotated / radius_x) ** 2 + (Y_rotated / radius_y) ** 2 <= 1

    return mask.astype(np.uint8) * 255
