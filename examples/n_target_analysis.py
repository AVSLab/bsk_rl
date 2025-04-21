import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# Enable interactive mode (for Matplotlib)
plt.ion()

# Ensure the 'plots' directory exists
os.makedirs("plots", exist_ok=True)

# Load the data
data_dir = "data"
inspector_sigmaBN = np.load(os.path.join(data_dir, "inspector_sigmaBN.npy"))[0]
inspector_r_BN_N = np.load(os.path.join(data_dir, "inspector_r_BN_N.npy"))[0]
inspector_omegaBN = np.load(os.path.join(data_dir, "inspector_omegaBN.npy"))[0]
sim_time = np.load(os.path.join(data_dir, "sim_time.npy"))

# Load all target data dynamically
target_files = sorted([f for f in os.listdir(data_dir) if f.startswith("target_r_BN_N_target_") and f.endswith(".npy")])
target_data = {int(f.split("_")[-1].split(".")[0]): np.load(os.path.join(data_dir, f))[0] for f in target_files}
target_data = dict(sorted(target_data.items()))  # Sort targets by index

# Compute pointing errors
pointing_errors = []
inspector_z_axes = []
angular_vel_magnitudes = np.linalg.norm(inspector_omegaBN, axis=1)
alignment_values = []
relative_vectors = []

num_targets = len(target_data)
switch_time = int(sim_time[-1] / num_targets)  # Assuming equal time per target

for i in range(len(inspector_sigmaBN)):
    sigma_BN = inspector_sigmaBN[i]
    rot_matrix = R.from_mrp(sigma_BN).as_matrix()
    inspector_z_axis = rot_matrix[:, 2]
    inspector_z_axes.append(inspector_z_axis)

    target_index = min(i // switch_time, num_targets - 1)
    target_r_BN_N = target_data[target_index]

    relative_vector = target_r_BN_N[i] - inspector_r_BN_N[i]
    relative_vectors.append(relative_vector)

    relative_vector_unit = relative_vector / np.linalg.norm(relative_vector)
    inspector_z_unit = inspector_z_axis / np.linalg.norm(inspector_z_axis)

    dot_product = np.clip(np.dot(relative_vector_unit, inspector_z_unit), -1.0, 1.0)
    error_angle = np.arccos(dot_product) * (180 / np.pi)
    alignment_values.append(error_angle)
    pointing_errors.append(error_angle)

inspector_z_axes = np.array(inspector_z_axes)
alignment_values = np.array(alignment_values)

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

# === Plot Pointing Error and Angular Velocity ===
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

# === 3D Trajectory Plot (Matplotlib) ===
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot inspector trajectory
ax.plot(inspector_r_BN_N[:, 0], inspector_r_BN_N[:, 1], inspector_r_BN_N[:, 2], label="Inspector Trajectory", color="blue")

# Plot target trajectories
colors = cm.get_cmap("tab10", num_targets)
for idx, (target_idx, target_r_BN_N) in enumerate(target_data.items()):
    ax.plot(target_r_BN_N[:, 0], target_r_BN_N[:, 1], target_r_BN_N[:, 2], label=f"Target {target_idx} Trajectory", color=colors(idx))



# Add Earth sphere
earth_radius = 6_400_000  # 6400 km in meters
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color="lightblue", alpha=0.3, edgecolors="gray")  # Semi-transparent Earth

# Set equal scaling
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_xlim(- max_range, max_range)
ax.set_ylim(- max_range, max_range)
ax.set_zlim(- max_range, max_range)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
plt.savefig("plots/3D_trajectories.pdf", format="pdf")
plt.show()

# === 3D Interactive Plot (Plotly) ===
fig = go.Figure()

# Inspector trajectory
fig.add_trace(go.Scatter3d(
    x=inspector_r_BN_N[:, 0],
    y=inspector_r_BN_N[:, 1],
    z=inspector_r_BN_N[:, 2],
    mode='lines',
    line=dict(color='blue', width=2),
    name='Inspector'
))

# Target trajectories
for idx, (target_idx, target_r_BN_N) in enumerate(target_data.items()):
    fig.add_trace(go.Scatter3d(
        x=target_r_BN_N[:, 0],
        y=target_r_BN_N[:, 1],
        z=target_r_BN_N[:, 2],
        mode='lines',
        line=dict(dash="dash", width=2),
        name=f'Target {target_idx}'
    ))

# Set equal aspect ratio
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[- max_range, max_range]),
        yaxis=dict(range=[- max_range, max_range]),
        zaxis=dict(range=[- max_range, max_range])
    ),
    title="Interactive 3D Trajectory",
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.show()
