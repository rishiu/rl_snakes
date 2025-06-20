import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from code.utils import calculate_normals


def plot_snake_on_image(image, snake_points, title="Snake on Image", normal_scale=5.0):
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
    ax.imshow(image, cmap="gray")
    ax.set_aspect("equal", adjustable="box")

    if (
        snake_points is not None
        and snake_points.shape[0] == 2
        and snake_points.shape[1] > 0
    ):
        # Append the first point to the end to close the loop for plotting
        x_plot = np.append(snake_points[0, :], snake_points[0, 0])
        y_plot = np.append(snake_points[1, :], snake_points[1, 0])
        ax.plot(x_plot, y_plot, "r-")  # 'r-' for red line

        # Calculate and plot normals
        vertex_normals, segment_normals = calculate_normals(snake_points)

        segment_midpoints = (np.roll(snake_points, 1, axis=1) + snake_points) / 2
        if vertex_normals.shape[1] > 0:
            ax.quiver(
                snake_points[0, :],
                snake_points[1, :],
                vertex_normals[0, :],
                vertex_normals[1, :],
                color="blue",
                units="xy",
                scale=1.0 / normal_scale,
                headwidth=3,
                headlength=4,
                width=0.002 * min(image.shape),
                zorder=2,
            )
            ax.quiver(
                segment_midpoints[0, :],
                segment_midpoints[1, :],
                segment_normals[0, :],
                segment_normals[1, :],
                color="green",
                units="xy",
                scale=1.0 / normal_scale,
                headwidth=3,
                headlength=4,
                width=0.002 * min(image.shape),
                zorder=2,
            )
    else:
        print("Warning: Invalid or empty snake_points provided to plot_snake_on_image.")

    ax.set_title(title)
    ax.axis("off")  # Turn off axis numbers and ticks
    plt.tight_layout()

    return fig


def visualize_gradient_fields(original_image, ext_energy_obj):
    """Visualizes the original image and various gradient fields.

    Args:
        original_image (np.ndarray): The initial image provided to GradientExternalEnergy.
        ext_energy_obj (GradientExternalEnergy): The GradientExternalEnergy object holding the gradients.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Image Gradient Fields", fontsize=16)

    # Original Image
    axs[0, 0].imshow(original_image, cmap="gray")
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # gx - First order gradient (x-direction)
    im_gx = axs[0, 1].imshow(ext_energy_obj.gx, cmap="RdBu_r")
    axs[0, 1].set_title("gx (Sobel X)")
    axs[0, 1].axis("off")
    fig.colorbar(im_gx, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # gy - First order gradient (y-direction)
    im_gy = axs[0, 2].imshow(ext_energy_obj.gy, cmap="RdBu_r")
    axs[0, 2].set_title("gy (Sobel Y)")
    axs[0, 2].axis("off")
    fig.colorbar(im_gy, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # Gradient Magnitude
    grad_mag = np.sqrt(ext_energy_obj.gx**2 + ext_energy_obj.gy**2)
    im_mag = axs[1, 0].imshow(grad_mag, cmap="viridis")
    axs[1, 0].set_title("Gradient Magnitude")
    axs[1, 0].axis("off")
    fig.colorbar(im_mag, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # gxx - Second order gradient (d(gx)/dx)
    im_gxx = axs[1, 1].imshow(ext_energy_obj.gxx, cmap="RdBu_r")
    axs[1, 1].set_title("gxx (d(gx)/dx)")
    axs[1, 1].axis("off")
    fig.colorbar(im_gxx, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # gyy - Second order gradient (d(gy)/dy)
    im_gyy = axs[1, 2].imshow(ext_energy_obj.gyy, cmap="RdBu_r")
    axs[1, 2].set_title("gyy (d(gy)/dy)")
    axs[1, 2].axis("off")
    fig.colorbar(im_gyy, ax=axs[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    return fig


def visualize_snake_evolution(
    image, snake_coords_list, title="Snake Evolution", normal_scale=5.0
):
    if not snake_coords_list:
        print("No snake coordinates to visualize.")
        plt.imshow(image, cmap="gray")
        plt.title("Image (No Snake Data)")
        plt.show()
        return

    fig, ax = plt.subplots()
    fig.suptitle(title)
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(image, cmap="gray")
    ax.set_aspect("equal", adjustable="box")

    initial_snake_points = snake_coords_list[0]
    x_plot_initial = np.append(initial_snake_points[0, :], initial_snake_points[0, 0])
    y_plot_initial = np.append(initial_snake_points[1, :], initial_snake_points[1, 0])

    (line,) = ax.plot(x_plot_initial, y_plot_initial, "r-")

    # Calculate and plot initial normals
    initial_normals = calculate_normals(initial_snake_points)[0]
    # quiver_normals = ax.quiver(initial_snake_points[0,:], initial_snake_points[1,:],
    #                            initial_normals[0,:], initial_normals[1,:],
    #                            color='blue', scale_units='xy', scale=1.0/normal_scale, width=0.005)
    # Simpler: use scale relative to axes units. `scale` in quiver means 1 unit along arrow = `scale` data units.
    # So a smaller `scale` value makes arrows longer if U,V are unit vectors.
    # We want arrows of length `normal_scale` in data units.
    # So if U,V are unit vectors, quiver(X,Y,U,V, scale_units='xy', scale=1/normal_scale)
    quiver_normals = ax.quiver(
        initial_snake_points[0, :],
        initial_snake_points[1, :],
        initial_normals[0, :],
        initial_normals[1, :],
        color="blue",
        units="xy",
        scale=1.0 / normal_scale,
        headwidth=3,
        headlength=4,
        width=0.002 * min(image.shape),
    )

    ax.set_title(f"Iteration: 0 / {len(snake_coords_list)-1}")

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(
        ax=ax_slider,
        label="Iteration",
        valmin=0,
        valmax=len(snake_coords_list) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        iteration = int(slider.val)
        current_snake_points = snake_coords_list[iteration]

        x_coords_plot = np.append(
            current_snake_points[0, :], current_snake_points[0, 0]
        )
        y_coords_plot = np.append(
            current_snake_points[1, :], current_snake_points[1, 0]
        )

        line.set_xdata(x_coords_plot)
        line.set_ydata(y_coords_plot)

        # Update normals
        current_normals = calculate_normals(current_snake_points)[0]
        quiver_normals.set_offsets(
            np.c_[current_snake_points[0, :], current_snake_points[1, :]]
        )
        quiver_normals.set_UVC(current_normals[0, :], current_normals[1, :])

        ax.set_title(f"Iteration: {iteration} / {len(snake_coords_list)-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def visualize_snake_evolution_with_energy_rewards(
    image,
    snake_coords_list,
    rewards_list,
    internal_energy_list=None,
    external_energy_list=None,
    title="Snake Evolution with Metrics",
    normal_scale=5.0,
):
    if not snake_coords_list:
        print("No snake coordinates to visualize.")
        plt.imshow(image, cmap="gray")
        plt.title("Image (No Snake Data)")
        plt.show()
        return

    num_iterations = len(snake_coords_list)
    iterations = np.arange(num_iterations)

    fig = plt.figure(figsize=(15, 12))  # Adjusted figure size for more plots
    # GridSpec: 3 rows for plots (snake, metrics, slider), 3 columns for metrics row
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1.5, 0.2], hspace=0.4, wspace=0.3)

    ax_snake = fig.add_subplot(
        gs[0, :]
    )  # Snake plot spans all columns in the first row
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_internal_e = fig.add_subplot(gs[1, 1])
    ax_external_e = fig.add_subplot(gs[1, 2])
    ax_slider_area = fig.add_subplot(
        gs[2, :]
    )  # Slider spans all columns in the last row

    fig.suptitle(title, fontsize=16)

    # --- Snake plot ---
    ax_snake.imshow(image, cmap="gray")
    ax_snake.set_aspect("equal", adjustable="box")

    initial_snake_points = snake_coords_list[0]
    x_plot_initial = np.append(initial_snake_points[0, :], initial_snake_points[0, 0])
    y_plot_initial = np.append(initial_snake_points[1, :], initial_snake_points[1, 0])

    (line,) = ax_snake.plot(x_plot_initial, y_plot_initial, "r-")

    initial_normals_data = calculate_normals(initial_snake_points)
    if isinstance(initial_normals_data, tuple):
        initial_normals = initial_normals_data[0]
    else:
        initial_normals = initial_normals_data

    quiver_normals = ax_snake.quiver(
        initial_snake_points[0, :],
        initial_snake_points[1, :],
        initial_normals[0, :],
        initial_normals[1, :],
        color="blue",
        units="xy",
        scale=1.0 / normal_scale,
        headwidth=3,
        headlength=4,
        width=0.002 * min(image.shape[:2]),
    )

    ax_snake.set_title(f"Iteration: 0 / {num_iterations-1}")

    # --- Metrics plots ---
    vline_reward, vline_internal_e, vline_external_e = None, None, None

    if rewards_list:
        ax_reward.plot(iterations, rewards_list, label="Reward", color="green")
        ax_reward.set_title("Reward vs. Iteration")
        ax_reward.set_xlabel("Iteration")
        ax_reward.set_ylabel("Reward")
        ax_reward.legend()
        ax_reward.grid(True)
        vline_reward = ax_reward.axvline(0, color="gray", linestyle="--", lw=1)

    if internal_energy_list:
        ax_internal_e.plot(
            iterations, internal_energy_list, label="Internal Energy", color="purple"
        )
        ax_internal_e.set_title("Internal Energy vs. Iteration")
        ax_internal_e.set_xlabel("Iteration")
        ax_internal_e.set_ylabel("Energy")
        ax_internal_e.legend()
        ax_internal_e.grid(True)
        vline_internal_e = ax_internal_e.axvline(0, color="gray", linestyle="--", lw=1)

    if external_energy_list:
        ax_external_e.plot(
            iterations, external_energy_list, label="External Energy", color="orange"
        )
        ax_external_e.set_title("External Energy vs. Iteration")
        ax_external_e.set_xlabel("Iteration")
        ax_external_e.set_ylabel("Energy")
        ax_external_e.legend()
        ax_external_e.grid(True)
        vline_external_e = ax_external_e.axvline(0, color="gray", linestyle="--", lw=1)

    # --- Slider ---
    slider = Slider(
        ax=ax_slider_area,
        label="Iteration",
        valmin=0,
        valmax=num_iterations - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        iteration = int(slider.val)
        current_snake_points = snake_coords_list[iteration]

        x_coords_plot = np.append(
            current_snake_points[0, :], current_snake_points[0, 0]
        )
        y_coords_plot = np.append(
            current_snake_points[1, :], current_snake_points[1, 0]
        )

        line.set_xdata(x_coords_plot)
        line.set_ydata(y_coords_plot)

        current_normals_data = calculate_normals(current_snake_points)
        if isinstance(current_normals_data, tuple):
            current_normals = current_normals_data[0]
        else:
            current_normals = current_normals_data

        quiver_normals.set_offsets(
            np.c_[current_snake_points[0, :], current_snake_points[1, :]]
        )
        quiver_normals.set_UVC(current_normals[0, :], current_normals[1, :])

        ax_snake.set_title(f"Iteration: {iteration} / {num_iterations-1}")

        # Update vertical lines on metrics plots
        if vline_reward:
            vline_reward.set_xdata([iteration])
        if vline_internal_e:
            vline_internal_e.set_xdata([iteration])
        if vline_external_e:
            vline_external_e.set_xdata([iteration])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Gridspec handles layout, adjust if necessary
    fig.tight_layout(
        rect=[0, 0.03, 1, 0.97]
    )  # Adjust rect to ensure suptitle and slider are visible
    plt.show()
