import numpy as np
from skimage.filters import sobel_h, sobel_v, gaussian
from code.utils import bilinear_interpolate


class GradientExternalEnergy:
    def __init__(self, img):
        self.image = img  # gaussian(img, sigma=1, preserve_range=True, channel_axis=None if img.ndim == 2 else -1)
        self.img_height, self.img_width = self.image.shape[:2]

        # First order gradients from the original image
        self.gx = sobel_v(self.image)
        self.gy = sobel_h(self.image)

        # Second order derivatives needed for F = -grad(E_ext) where E_ext = -(gx^2 + gy^2)
        # F_x = d/dx (gx^2 + gy^2) = 2*gx*gxx + 2*gy*gyx
        # F_y = d/dy (gx^2 + gy^2) = 2*gx*gxy + 2*gy*gyy
        self.gxx = sobel_v(self.gx)  # d(gx)/dx
        self.gxy = sobel_h(self.gx)  # d(gx)/dy
        self.gyx = sobel_v(self.gy)  # d(gy)/dx (should be similar to gxy)
        self.gyy = sobel_h(self.gy)  # d(gy)/dy

    def get_energy(self, snake_points):
        gx_vals = bilinear_interpolate(self.gx, snake_points)
        gy_vals = bilinear_interpolate(self.gy, snake_points)
        return np.sum(np.power(gx_vals, 2) + np.power(gy_vals, 2))

    def __call__(self, snake_points):
        gx_vals = bilinear_interpolate(self.gx, snake_points)
        gy_vals = bilinear_interpolate(self.gy, snake_points)
        gxx_vals = bilinear_interpolate(self.gxx, snake_points)
        gxy_vals = bilinear_interpolate(self.gxy, snake_points)
        gyx_vals = bilinear_interpolate(self.gyx, snake_points)
        gyy_vals = bilinear_interpolate(self.gyy, snake_points)

        force_x = -(2 * gx_vals * gxx_vals + 2 * gy_vals * gyx_vals)
        force_y = -(2 * gx_vals * gxy_vals + 2 * gy_vals * gyy_vals)
        return force_x, force_y
