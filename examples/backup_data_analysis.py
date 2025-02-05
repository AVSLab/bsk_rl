import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
# from scipy.special import label
from stack_data import markers_from_ranges
from skimage.measure import label
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Ensure the 'plots' directory exists
os.makedirs("plots", exist_ok=True)

# Load the data
data_dir = "data"
inspector_sigmaBN = np.load(os.path.join(data_dir, "inspector_sigmaBN.npy"))
inspector_r_BN_N = np.load(os.path.join(data_dir, "inspector_r_BN_N.npy"))
target_r_BN_N = np.load(os.path.join(data_dir, "target_r_BN_N.npy"))
inspector_omegaBN = np.load(os.path.join(data_dir, "inspector_omegaBN.npy"))

# Compute the vector from inspector to target
relative_vector = np.array(target_r_BN_N) - np.array(inspector_r_BN_N)

# Compute pointing error and extract inspector Z-axis direction
pointing_errors = []
inspector_z_axes = []
angular_vel_magnitudes = np.linalg.norm(inspector_omegaBN, axis=1)  # Compute magnitude of omega_BN
alignment_values = []  # Store dot product values for colormap

for i in range(len(inspector_sigmaBN)):
    sigma_BN = inspector_sigmaBN[i]
    # Convert Modified Rodrigues Parameters (MRP) to a rotation matrix
    rot_matrix = R.from_mrp(sigma_BN).as_matrix()
    inspector_z_axis = rot_matrix[:, 2]  # Z-axis is the third column
    inspector_z_axes.append(inspector_z_axis)

    # Normalize vectors
    relative_vector_unit = relative_vector[i] / np.linalg.norm(relative_vector[i])
    inspector_z_unit = inspector_z_axis / np.linalg.norm(inspector_z_axis)

    # Compute pointing error (angle between vectors in degrees)
    dot_product = np.clip(np.dot(relative_vector_unit, inspector_z_unit), -1.0, 1.0)
    error_angle = np.arccos(dot_product) * (180 / np.pi)  # Convert to degrees
    # alignment_values.append(dot_product)
    alignment_values.append(dot_product)
    pointing_errors.append(error_angle)

# Convert lists to numpy arrays for plotting
inspector_z_axes = np.array(inspector_z_axes)
alignment_values = np.array(alignment_values)

### Plot 1: Pointing Error and Angular Velocity ###
fig, ax1 = plt.subplots(figsize=(10, 5))

# Primary Y-axis: Pointing error
ax1.plot(pointing_errors, label="Pointing Error (degrees)", color="red")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Pointing Error (degrees)", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid()

# Secondary Y-axis: Angular velocity magnitude
ax2 = ax1.twinx()
ax2.plot(angular_vel_magnitudes, label="Angular Velocity (rad/s)", color="blue", linestyle="dashed")
ax2.set_yscale("log")
ax2.set_ylabel("Angular Velocity (rad/s)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

fig.suptitle("Inspector Satellite Pointing Error and Angular Velocity Over Time")
fig.legend(loc="upper right")
plt.savefig("plots/backup_pointing_error.pdf", format="pdf")  # Save as PDF
plt.show()

# # ### Plot 2: 3D Trajectories with Orientation Arrows ###
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot spacecraft trajectories
# ax.plot(inspector_r_BN_N[:, 0], inspector_r_BN_N[:, 1], inspector_r_BN_N[:, 2], label="Inspector Trajectory", color="blue")
# ax.plot(target_r_BN_N[:, 0], target_r_BN_N[:, 1], target_r_BN_N[:, 2], label="Target Trajectory", color="green")
#
# # Plot arrows every 20 data points
# step = 8
# arrow_scale = 1e6  # Scale factor for arrows
# c_target = "purple"
# ax.plot(target_r_BN_N[0, 0], target_r_BN_N[0, 1], target_r_BN_N[0, 2], label=str("Target Trajectory ~"+str(step)+" min interval"), color=c_target,marker='x')
#
# for i in range(0, int(len(inspector_r_BN_N)/1), step):
#     # Inspector position
#     x, y, z = inspector_r_BN_N[i]
#     ax.plot(target_r_BN_N[i, 0], target_r_BN_N[i, 1], target_r_BN_N[i, 2], color=c_target,marker='x')
#
#     # Vector from inspector to target (black arrow)
#     relative_vector_unit = relative_vector[i] / np.linalg.norm(relative_vector[i])
#     dx, dy, dz = relative_vector_unit * 3
#     ax.quiver(x, y, z, dx, dy, dz, color="black", length=arrow_scale, linewidth=1, arrow_length_ratio=0.3,label="Inspector to Target" if i == 0 else "")
#
#     # Inspector Z-axis direction (red arrow)
#     zx, zy, zz = inspector_z_axes[i] * 2  # Scale z-axis direction
#     ax.quiver(x, y, z, zx, zy, zz, color="orange", length=arrow_scale, linewidth=1, arrow_length_ratio=0.3,label="Bore-sight direction" if i == 0 else "")
# ax.plot(target_r_BN_N[0, 0], target_r_BN_N[0, 1], target_r_BN_N[0, 2], color=c_target,marker='x')
# Define colormap for alignment
norm = mcolors.Normalize(vmin=min(alignment_values)*0.8, vmax=1.0)  # Normalize between -1 and 1
norm = mcolors.Normalize(vmin=0, vmax=max(alignment_values))  # Normalize between 0 and max error angle

cmap = cm.get_cmap("inferno_r")  # or use plasma or inferno add a _r at the end of the name to reverse it Use coolwarm colormap

### Plot 2: 3D Trajectories with Orientation Arrows ###
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot spacecraft trajectories
ax.plot(inspector_r_BN_N[:, 0], inspector_r_BN_N[:, 1], inspector_r_BN_N[:, 2], label="Inspector Trajectory", color="blue")
ax.plot(target_r_BN_N[:, 0], target_r_BN_N[:, 1], target_r_BN_N[:, 2], label="Target Trajectory", color="green")

# Plot arrows every 'step' data points
step = 8
arrow_scale = 1e6  # Scale factor for arrows
c_target = "purple"
ax.plot(target_r_BN_N[0, 0], target_r_BN_N[0, 1], target_r_BN_N[0, 2], label=str("Target Trajectory ~"+str(step)+" min interval"), color=c_target,marker='x')

for i in range(0, int(len(inspector_r_BN_N)/1), step):
    # Inspector position
    x, y, z = inspector_r_BN_N[i]

    # Target position
    ax.plot(target_r_BN_N[i, 0], target_r_BN_N[i, 1], target_r_BN_N[i, 2], color=c_target,marker='x')

    # Vector from inspector to target (black arrow)
    relative_vector_unit = relative_vector[i] / np.linalg.norm(relative_vector[i])
    dx, dy, dz = relative_vector_unit * 3
    ax.quiver(x, y, z, dx, dy, dz, color="grey", length=arrow_scale, linewidth=1, arrow_length_ratio=0.2,
              label="Inspector to Target" if i == 0 else "")

    # Inspector Z-axis direction colored by alignment value
    zx, zy, zz = inspector_z_axes[i] * 2  # Scale z-axis direction
    color = cmap(norm(alignment_values[i]))
    ax.quiver(x, y, z, zx, zy, zz, color=color, length=arrow_scale, linewidth=1, arrow_length_ratio=0.25,
              label="Bore-sight direction" if i == 0 else "")

# # Add colorbar
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Empty array for colorbar
# cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
# cbar.set_label("Alignment (Dot Product)")

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array for colorbar
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Pointing Error (degrees)")

# Labels and legend
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("3D Trajectories with Orientation Arrows")
ax.legend()
plt.savefig("plots/backup_3D_trajectories.pdf", format="pdf")  # Save as PDF
plt.show()
