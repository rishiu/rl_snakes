import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def visualize_snake_evolution(image, snake_coords_list, title="Snake Evolution"):
    if not snake_coords_list:
        print("No snake coordinates to visualize.")
        plt.imshow(image, cmap='gray')
        plt.title("Image (No Snake Data)")
        plt.show()
        return

    fig, ax = plt.subplots()
    fig.suptitle(title)
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(image, cmap='gray')
    
    initial_snake_points = snake_coords_list[0]
    x_plot_initial = np.append(initial_snake_points[0,:], initial_snake_points[0,0])
    y_plot_initial = np.append(initial_snake_points[1,:], initial_snake_points[1,0])
    
    line, = ax.plot(x_plot_initial, y_plot_initial, 'r-')

    ax.set_title(f"Iteration: 0 / {len(snake_coords_list)-1}")

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax=ax_slider,
        label='Iteration',
        valmin=0,
        valmax=len(snake_coords_list) - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        iteration = int(slider.val)
        current_snake_points = snake_coords_list[iteration]
        
        x_coords_plot = np.append(current_snake_points[0,:], current_snake_points[0,0])
        y_coords_plot = np.append(current_snake_points[1,:], current_snake_points[1,0])
        
        line.set_xdata(x_coords_plot)
        line.set_ydata(y_coords_plot)
        ax.set_title(f"Iteration: {iteration} / {len(snake_coords_list)-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def visualize_gradient_fields(original_image, ext_energy_obj):
    """Visualizes the original image and various gradient fields.

    Args:
        original_image (np.ndarray): The initial image provided to GradientExternalEnergy.
        ext_energy_obj (GradientExternalEnergy): The GradientExternalEnergy object holding the gradients.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Gradient Fields', fontsize=16)

    # Original Image
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # gx - First order gradient (x-direction)
    im_gx = axs[0, 1].imshow(ext_energy_obj.gx, cmap='RdBu_r')
    axs[0, 1].set_title('gx (Sobel X)')
    axs[0, 1].axis('off')
    fig.colorbar(im_gx, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # gy - First order gradient (y-direction)
    im_gy = axs[0, 2].imshow(ext_energy_obj.gy, cmap='RdBu_r')
    axs[0, 2].set_title('gy (Sobel Y)')
    axs[0, 2].axis('off')
    fig.colorbar(im_gy, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # Gradient Magnitude
    grad_mag = np.sqrt(ext_energy_obj.gx**2 + ext_energy_obj.gy**2)
    im_mag = axs[1, 0].imshow(grad_mag, cmap='viridis')
    axs[1, 0].set_title('Gradient Magnitude')
    axs[1, 0].axis('off')
    fig.colorbar(im_mag, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # gxx - Second order gradient (d(gx)/dx)
    im_gxx = axs[1, 1].imshow(ext_energy_obj.gxx, cmap='RdBu_r')
    axs[1, 1].set_title('gxx (d(gx)/dx)')
    axs[1, 1].axis('off')
    fig.colorbar(im_gxx, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # gyy - Second order gradient (d(gy)/dy)
    im_gyy = axs[1, 2].imshow(ext_energy_obj.gyy, cmap='RdBu_r')
    axs[1, 2].set_title('gyy (d(gy)/dy)')
    axs[1, 2].axis('off')
    fig.colorbar(im_gyy, ax=axs[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def plot_snake_on_image(image, snake_points, title="Snake on Image"):
    """Plots a single snake contour on a given image and returns the figure.

    Args:
        image (np.ndarray): The background image.
        snake_points (np.ndarray): A numpy array of shape (2, N), representing N 2D points
                                   (row 0 for x, row 1 for y).
        title (str, optional): The title for the plot. Defaults to "Snake on Image".

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object with the plot.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    
    if snake_points is not None and snake_points.shape[0] == 2 and snake_points.shape[1] > 0:
        # Append the first point to the end to close the loop for plotting
        x_plot = np.append(snake_points[0,:], snake_points[0,0])
        y_plot = np.append(snake_points[1,:], snake_points[1,0])
        ax.plot(x_plot, y_plot, 'r-') # 'r-' for red line
    else:
        print("Warning: Invalid or empty snake_points provided to plot_snake_on_image.")

    ax.set_title(title)
    ax.axis('off') # Turn off axis numbers and ticks
    plt.tight_layout()
    
    return fig