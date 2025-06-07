import numpy as np
import matplotlib.pyplot as plt

def ma(arr, idx):
    return arr[idx % len(arr)]

def calculate_normals(points):
    point_centroid = np.mean(points, axis=1)

    shifted_pts = np.roll(points, 1, axis=1)
    segment_dir = shifted_pts - points

    centroid_vec = point_centroid[:, np.newaxis] - points

    opt_one = np.stack((-segment_dir[1,:], segment_dir[0,:]), axis=0)

    opt_two = np.stack((segment_dir[1,:], -segment_dir[0,:]), axis=0)

    opt_one_dot = np.sum(opt_one * centroid_vec, axis=0) < 0
    opt_two_dot = np.sum(opt_two * centroid_vec, axis=0) < 0

    segment_normals = opt_one_dot * opt_one + opt_two_dot * opt_two

    shifted_normals = np.roll(segment_normals, -1, axis=1)
    vertex_normals = (segment_normals + shifted_normals) / 2

    segment_midpoints = (points + shifted_pts) / 2.0

    epsilon = 1e-8  # Small number to handle near-zero magnitudes safely

    seg_norm_magnitudes = np.linalg.norm(segment_normals, axis=0, keepdims=True)
    
    is_seg_norm_effectively_zero = seg_norm_magnitudes < epsilon
    
    safe_seg_norm_magnitudes = np.where(is_seg_norm_effectively_zero, 1.0, seg_norm_magnitudes)
    plot_segment_normal_directions = segment_normals / safe_seg_norm_magnitudes
    
    plot_segment_normal_directions[0, is_seg_norm_effectively_zero[0,:]] = 0.0
    plot_segment_normal_directions[1, is_seg_norm_effectively_zero[0,:]] = 0.0

    vert_norm_magnitudes = np.linalg.norm(vertex_normals, axis=0, keepdims=True)
    is_vert_norm_effectively_zero = vert_norm_magnitudes < epsilon
    safe_vert_norm_magnitudes = np.where(is_vert_norm_effectively_zero, 1.0, vert_norm_magnitudes)
    
    plot_vertex_normal_directions = vertex_normals / safe_vert_norm_magnitudes
    plot_vertex_normal_directions[0, is_vert_norm_effectively_zero[0,:]] = 0.0
    plot_vertex_normal_directions[1, is_vert_norm_effectively_zero[0,:]] = 0.0

    return plot_vertex_normal_directions, plot_segment_normal_directions

def bilinear_interpolate(image, points):
    if image.ndim != 2:
        raise ValueError("Image must be a 2D array.")
    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError("Points must be a 2D array of shape (2, N).")

    x = points[0, :]
    y = points[1, :]

    img_height, img_width = image.shape

    # Floor and ceil coordinates for interpolation
    x_floor = np.floor(x).astype(int)
    y_floor = np.floor(y).astype(int)
    x_ceil = np.ceil(x).astype(int)
    y_ceil = np.ceil(y).astype(int)

    # Clip coordinates to be within image boundaries
    x_floor_c = np.clip(x_floor, 0, img_width - 1)
    y_floor_c = np.clip(y_floor, 0, img_height - 1)
    x_ceil_c = np.clip(x_ceil, 0, img_width - 1)
    y_ceil_c = np.clip(y_ceil, 0, img_height - 1)

    # Deltas for interpolation
    dx = x - x_floor
    dy = y - y_floor

    # Values at the four corners
    # f(y,x)
    f00 = image[y_floor_c, x_floor_c]  # Value at (x_floor, y_floor)
    f10 = image[y_floor_c, x_ceil_c]   # Value at (x_ceil, y_floor)
    f01 = image[y_ceil_c, x_floor_c]   # Value at (x_floor, y_ceil)
    f11 = image[y_ceil_c, x_ceil_c]    # Value at (x_ceil, y_ceil)

    # Bilinear interpolation formula
    interpolated_values = (f00 * (1 - dx) * (1 - dy) +
                           f10 * dx * (1 - dy) +
                           f01 * (1 - dx) * dy +
                           f11 * dx * dy)

    return interpolated_values


def to_banded_format(A, l, u):
    N = A.shape[0]
    ab = np.zeros((l + u + 1, N))

    ab[u, :] = np.diag(A, k=0)

    for i in range(1, u + 1):
        diag_elements = np.diag(A, k=i)
        ab[u - i, i:] = diag_elements
    
    for i in range(1, l + 1):
        diag_elements = np.diag(A, k=-i)
        ab[u + i, :N - i] = diag_elements
        
    return ab

def create_circle_points(num_points, center_x=0, center_y=0, radius=1):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_points = center_x + radius * np.cos(angles)
    y_points = center_y + radius * np.sin(angles)
    return np.array([x_points, y_points])

def create_ellipse_points(num_points, center_x=0, center_y=0, radius_x=1, radius_y=1, rotation_angle_rad=0):
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    x_unrotated = radius_x * np.cos(t)
    y_unrotated = radius_y * np.sin(t)
    
    cos_phi = np.cos(rotation_angle_rad)
    sin_phi = np.sin(rotation_angle_rad)
    
    x_rotated = x_unrotated * cos_phi - y_unrotated * sin_phi
    y_rotated = x_unrotated * sin_phi + y_unrotated * cos_phi
    
    x_points = center_x + x_rotated
    y_points = center_y + y_rotated
    
    return np.array([x_points, y_points])