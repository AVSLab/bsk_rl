import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm
import matplotlib.colors as mcolors



# Ensure the 'plots' directory exists
os.makedirs("plots", exist_ok=True)

# Load the data
data_dir = "2targets_2000sec_data"
data_dir = "data"
inspector_sigmaBN = np.load(os.path.join(data_dir, "inspector_sigmaBN.npy"))[0]
inspector_r_BN_N = np.load(os.path.join(data_dir, "inspector_r_BN_N.npy"))[0] #.tolist()
inspector_omegaBN = np.load(os.path.join(data_dir, "inspector_omegaBN.npy"))[0] #.tolist()
sim_time = np.load(os.path.join(data_dir, "sim_time.npy"))

# # Load all target data dynamically
# target_files = sorted([f for f in os.listdir(data_dir) if f.startswith("target_r_BN_N_target_") and f.endswith(".npy")])
# # target_data = {file: np.load(os.path.join(data_dir, file)) for file in target_files}
# target_data = {file: np.array(np.load(os.path.join(data_dir, file))) for file in target_files}

I_r_BN_N = inspector_r_BN_N.tolist()
target0_r_BN_N = np.load(os.path.join(data_dir, "target_r_BN_N_target_0.npy"))[0] #.tolist()
target1_r_BN_N = np.load(os.path.join(data_dir, "target_r_BN_N_target_1.npy"))[0] #.tolist()

# Compute pointing errors
pointing_errors = []
inspector_z_axes = []
angular_vel_magnitudes = np.linalg.norm(inspector_omegaBN, axis=1)  # Compute magnitude of omega_BN
alignment_values = []
relative_vector=[]

for i in range(len(inspector_sigmaBN)):
    sigma_BN = inspector_sigmaBN[i]
    rot_matrix = R.from_mrp(sigma_BN).as_matrix()
    inspector_z_axis = rot_matrix[:, 2]
    inspector_z_axes.append(inspector_z_axis)

    # Compute pointing error with respect to the current target
    if i < 2000/2: #sim_time[-1]/n_targets:

        relative_vector.append(target0_r_BN_N[i] - inspector_r_BN_N[i])
    else:
        relative_vector.append(target1_r_BN_N[i] - inspector_r_BN_N[i])

    relative_vector_unit = relative_vector[-1] / np.linalg.norm(relative_vector[-1])
    inspector_z_unit = inspector_z_axis / np.linalg.norm(inspector_z_axis)

    dot_product = np.clip(np.dot(relative_vector_unit, inspector_z_unit), -1.0, 1.0)
    error_angle = np.arccos(float(dot_product)) * (180 / np.pi)
    alignment_values.append(error_angle)
    pointing_errors.append(error_angle)

inspector_z_axes = np.array(inspector_z_axes)
alignment_values = np.array(alignment_values)

# Define colormap for alignment based on pointing error
norm = mcolors.Normalize(vmin=0, vmax=max(alignment_values)*1.3)  # Normalize between 0 and max error angle
cmap = plt.get_cmap("inferno")  # or use plasma or inferno

# Plot Pointing Error and Angular Velocity
fig, ax1 = plt.subplots(figsize=(10, 5))
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
plt.savefig("plots/pointing_error.pdf", format="pdf")
plt.show()

# # 3D Trajectory Plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(inspector_r_BN_N[:, 0], inspector_r_BN_N[:, 1], inspector_r_BN_N[:, 2], label="Inspector Trajectory", color="blue")
# colors = cm.get_cmap("tab10", len(target_data))
# for idx, (file, target_r_BN_N) in enumerate(target_data.items()):
#     ax.plot(target_r_BN_N[:, 0], target_r_BN_N[:, 1], target_r_BN_N[:, 2], label=f"{file}", color=colors(idx))
# ax.legend()
# plt.savefig("plots/3D_trajectories.pdf", format="pdf")
# plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot spacecraft trajectories
c_target1 = "purple"
c_target2 = "green"
ax.plot(inspector_r_BN_N[:, 0], inspector_r_BN_N[:, 1], inspector_r_BN_N[:, 2], label="Inspector Trajectory", color="blue")
# ax.plot(target0_r_BN_N[:, 0], target0_r_BN_N[:, 1], target0_r_BN_N[:, 2], label="Target0 Trajectory", color="green")
# ax.plot(target1_r_BN_N[:, 0], target1_r_BN_N[:, 1], target1_r_BN_N[:, 2], label="Target1 Trajectory", color="purple")

# Plot arrows every 8 data points
step = 120
arrow_scale = 1e6  # Scale factor for arrows
target_orbit= 'GTO'
if target_orbit == 'GEO':
    arrow_scale = 3e6
elif target_orbit == 'GTO':
    arrow_scale = 2e6

ax.scatter(target0_r_BN_N[0, 0], target0_r_BN_N[0, 1], target0_r_BN_N[0, 2], label=str("Target0 Trajectory ~"+str(step)+" sec interval"), color=c_target1,marker='x')
ax.scatter(target1_r_BN_N[0, 0], target1_r_BN_N[0, 1], target1_r_BN_N[0, 2], label=str("Target1 Trajectory ~"+str(step)+" sec interval"), color=c_target2,marker='x')

for i in range(0, int(len(inspector_r_BN_N)/1), step):
    # Inspector position
    x, y, z = inspector_r_BN_N[i]

    # Target position
    ax.scatter(target0_r_BN_N[i, 0], target0_r_BN_N[i, 1], target0_r_BN_N[i, 2], color=c_target1,marker='x')
    ax.scatter(target1_r_BN_N[i, 0], target1_r_BN_N[i, 1], target1_r_BN_N[i, 2], color=c_target2,marker='x')

    # Vector from inspector to target (black arrow)
    relative_vector_unit = relative_vector[i] / np.linalg.norm(relative_vector[i])
    dx, dy, dz = relative_vector_unit * 3
    ax.quiver(x, y, z, dx, dy, dz, color="grey", length=arrow_scale, linewidth=1, arrow_length_ratio=0.2,
              label="Inspector to Target" if i == 0 else "")

    # Inspector Z-axis direction colored by pointing error
    zx, zy, zz = inspector_z_axes[i] * 2.3  # Scale z-axis direction
    color = cmap(norm(alignment_values[i]))
    ax.quiver(x, y, z, zx, zy, zz, color=color, length=arrow_scale, linewidth=1, arrow_length_ratio=0.25,
              label="Bore-sight direction" if i == 0 else "")

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array for colorbar
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Pointing Error (degrees)")

# Labels and legend
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
# ax.set_title("3D Trajectories with Orientation Arrows")
ax.legend()
plt.savefig("plots/3D_trajectories.pdf", format="pdf")  # Save as PDF
plt.show()