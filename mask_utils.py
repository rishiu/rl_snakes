import numpy as np
import random

def create_circular_mask(height, width, center_x=None, center_y=None, radius=None):
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    if radius is None:
        radius = min(height, width) // 4

    Y, X = np.ogrid[:height, :width]
    dist_from_center_sq = (X - center_x)**2 + (Y - center_y)**2
    mask = dist_from_center_sq <= radius**2
    return mask.astype(np.uint8) * 255

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
        actual_cx_min = max(cx_min_constrained_by_boundary, cx_min_constrained_by_center)
        actual_cx_max = min(cx_max_constrained_by_boundary, cx_max_constrained_by_center)

        # Determine bounds for center_y
        # Constraint from image center
        cy_min_constrained_by_center = img_center_y - center_constraint_pixels
        cy_max_constrained_by_center = img_center_y + center_constraint_pixels

        # Constraint from image boundaries
        cy_min_constrained_by_boundary = radius
        cy_max_constrained_by_boundary = height - radius - 1
        
        # Combined constraints for center_y
        actual_cy_min = max(cy_min_constrained_by_boundary, cy_min_constrained_by_center)
        actual_cy_max = min(cy_max_constrained_by_boundary, cy_max_constrained_by_center)

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
        dist_from_center_sq = (X - center_x)**2 + (Y - center_y)**2
        circle_mask = dist_from_center_sq <= radius**2
        final_mask = np.logical_or(final_mask, circle_mask)
        
    return final_mask.astype(np.uint8) * 255