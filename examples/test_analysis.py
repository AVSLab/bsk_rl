import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Ensure the 'plots' directory exists
os.makedirs("plots", exist_ok=True)

# Load the data
data_dir = "data"
inspector_sigmaBN = np.load(os.path.join(data_dir, "inspector_sigmaBN.npy"))[0]
inspector_r_BN_N = np.load(os.path.join(data_dir, "inspector_r_BN_N.npy"))[0]
current_target_r_BN_N = np.load(os.path.join(data_dir, "currentTarget_r_BN_N.npy"))[0]
inspector_omegaBN = np.load(os.path.join(data_dir, "inspector_omegaBN.npy"))[0]
sim_time = np.load(os.path.join(data_dir, "sim_time.npy"))

# Dynamically load all target data
target_files = sorted([f for f in os.listdir(data_dir) if f.startswith("target_r_BN_N_target_") and f.endswith(".npy")])
target_data = {int(f.split("_")[-1].split(".")[0]): np.load(os.path.join(data_dir, f))[0] for f in target_files}

# Sort targets by index
target_data = dict(sorted(target_data.items()))
print("Number of Target files collected: ", len(target_files), ", target_data length: ",len(target_data))


# Compute pointing errors
pointing_errors = []
inspector_z_axes = []
angular_vel_magnitudes = np.linalg.norm(inspector_omegaBN, axis=1)
alignment_values = []
relative_vectors = []
current_target=[]
num_targets = len(target_data)

switch_time = int(sim_time[-1] / num_targets)  # Assuming equal time for each target
print('switch times', switch_time)
for i in range(int(len(inspector_sigmaBN))):
    sigma_BN = inspector_sigmaBN[i]
    rot_matrix = R.from_mrp(sigma_BN).as_matrix()
    inspector_z_axis = rot_matrix[:, 2]
    inspector_z_axes.append(inspector_z_axis)

    # Select correct target dynamically
    target_index = min(i // switch_time, num_targets - 1)
    target_r_BN_N = target_data[target_index]

    relative_vector = target_r_BN_N[i] - inspector_r_BN_N[i]
    current_target.append(target_r_BN_N[i])
    relative_vectors.append(relative_vector)

    relative_vector_unit = relative_vector / np.linalg.norm(relative_vector)
    inspector_z_unit = inspector_z_axis / np.linalg.norm(inspector_z_axis)

    dot_product = np.clip(np.dot(relative_vector_unit, inspector_z_unit), -1.0, 1.0)
    error_angle = np.arccos(dot_product) * (180 / np.pi)
    alignment_values.append(error_angle)
    pointing_errors.append(error_angle)

inspector_z_axes = np.array(inspector_z_axes)
alignment_values = np.array(alignment_values)

# Define colormap for alignment based on pointing error
norm = mcolors.Normalize(vmin=0, vmax=max(alignment_values) * 1.3)
cmap = plt.get_cmap("inferno")

# Disable interactive mode (fixing blank plot issue)
plt.ioff()

# Plot Pointing Error and Angular Velocity
fig, ax1 = plt.subplots(figsize=(20, 8))
ax1.plot(pointing_errors, label="Pointing Error (degrees)", color="red")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Pointing Error (degrees)", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(angular_vel_magnitudes, label="Angular Velocity (rad/s)", color="blue", linestyle="dashed")
ax2.set_yscale("log")
ax2.set_ylabel("Angular Velocity (rad/s)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

fig.suptitle("Inspector Satellite Pointing Error and Angular Velocity Over Time")
fig.legend(loc="upper right")

# Save and explicitly close
plt.savefig("plots/pointing_error.pdf", format="pdf")
plt.show()
plt.close(fig)  # Ensures figure is cleared for next plot

#
# 3D Trajectory Plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot inspector trajectory
ax.plot(inspector_r_BN_N[:, 0], inspector_r_BN_N[:, 1], inspector_r_BN_N[:, 2], label="Inspector Trajectory", color="blue")
ax.plot(inspector_r_BN_N[0, 0], inspector_r_BN_N[0, 1], inspector_r_BN_N[0, 2], marker = "X", color="blue")


# Plot all targets
colors = cm.get_cmap("tab10", num_targets)
for idx, (target_idx, target_r_BN_N) in enumerate(target_data.items()):
    if len(target_files) < 20:
        ax.plot(target_r_BN_N[:, 0], target_r_BN_N[:, 1], target_r_BN_N[:, 2], label=f"Target {target_idx} Trajectory", color=colors(idx), alpha=0.45) # , label=f"Target {target_idx} Trajectory"
    else:
        ax.plot(target_r_BN_N[:, 0], target_r_BN_N[:, 1], target_r_BN_N[:, 2], color=colors(idx), alpha=0.35) # , label=f"Target {target_idx} Trajectory"

# Add Earth sphere
earth_radius = 6_400_000  # 6400 km in meters
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 20)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

# ax.plot_surface(x, y, z, color="lightblue", alpha=0.00001, edgecolors="lightgray")  # Semi-transparent Earth

# === Find Plot Scaling Bounds ===
all_positions = [inspector_r_BN_N] + list(target_data.values())
max_range = 0
mid_x = mid_y = mid_z = 0

for pos in all_positions:
    x_limits = [pos[:, 0].min(), pos[:, 0].max()]
    y_limits = [pos[:, 1].min(), pos[:, 1].max()]
    z_limits = [pos[:, 2].min(), pos[:, 2].max()]

    max_range = max(max_range, (x_limits[1] - x_limits[0]) / 2.0,
                               (y_limits[1] - y_limits[0]) / 2.0,
                               (z_limits[1] - z_limits[0]) / 2.0)

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)


# Plot orientation arrows
step = 150
arrow_scale = 2.5e6  # Scale factor for arrows
target_orbit = 'GTO'
if target_orbit == 'GEO':
    arrow_scale = 3e6

# Plot target positions
# for target_idx, target_r_BN_N in target_data.items():
#     ax.scatter(target_r_BN_N[0, 0], target_r_BN_N[0, 1], target_r_BN_N[0, 2], label=f"Target {target_idx}", marker='X')
color_list=[]
for idx, (target_idx, target_r_BN_N) in enumerate(target_data.items()):
    ax.scatter(target_r_BN_N[0, 0], target_r_BN_N[0, 1], target_r_BN_N[0, 2], color=colors(idx) , marker='o') # , label=f"Target {target_idx} Start"
    color_list.append(colors(idx))

ax.scatter(current_target[0][0], current_target[0][1], current_target[0][2],  marker='o', color='black' , alpha=0.9, label ="Current Target")

for i in range(0, len(inspector_r_BN_N), step):
    x, y, z = inspector_r_BN_N[i]

    # Plot target positions
    if i != 0: #to not plot over the initial marking of the start of each trajectory
        for target_idx, target_r_BN_N in target_data.items():
            ax.scatter(target_r_BN_N[i, 0], target_r_BN_N[i, 1], target_r_BN_N[i, 2],  marker='x', color=colors(target_idx) , alpha=0.06)
        ax.scatter(current_target[i][0], current_target[i][1], current_target[i][2],  marker='x', color='black' , alpha=0.7)
    # Vector from inspector to target
    relative_vector_unit = relative_vectors[i] / np.linalg.norm(relative_vectors[i])
    dx, dy, dz = relative_vector_unit * 3
    # ax.quiver(x, y, z, dx, dy, dz, color="grey", length=arrow_scale, linewidth=1, arrow_length_ratio=0.2, alpha=0.6,
    #           label="Inspector to Target" if i == 0 else "")

    # Inspector Z-axis direction colored by pointing error
    zx, zy, zz = inspector_z_axes[i] * 2.3
    color = cmap(norm(alignment_values[i]))
    # ax.quiver(x, y, z, zx, zy, zz, color=color, length=arrow_scale, linewidth=1, arrow_length_ratio=0.25, alpha=0.85,
    #           label="Bore-sight direction" if i == 0 else "")

# for idx, (target_idx, target_r_BN_N) in enumerate(target_data.items()):
#     ax.scatter(target_r_BN_N[0, 0], target_r_BN_N[0, 1], target_r_BN_N[0, 2], label=f"Target {target_idx}", color=colors(idx) , marker='o')

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Pointing Error (degrees)")
max_range= max_range*4
# ax.set_xlim(-max_range, max_range)
# ax.set_ylim(- max_range, max_range)
# ax.set_zlim(- max_range, max_range)
# ax.set_box_aspect([1, 1, 1])

# Labels and legend
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.legend(loc='upper left')
plt.savefig("plots/3D_trajectories.pdf", format="pdf")
plt.show()
