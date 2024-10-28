import deepxde as dde
from deepxde.backend import tf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_domain = [-5.0, 5.0]
t_domain = [0.0, np.pi/2.0]

x = np.linspace(x_domain[0], x_domain[1], 256)
t = np.linspace(t_domain[0], t_domain[1], 201)
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

space_domain = dde.geometry.Interval(x_domain[0], x_domain[1])
time_domain = dde.geometry.TimeDomain(t_domain[0], t_domain[1])
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

def pde(x, y):
    # INPUTS:
    #     x: x[:,0] is x-coordinate
    #        x[:,1] is t-coordinate
    #     y: y[:,0] is the output's real part (u)
    #        y[:,1] is the output's imaginary part (v)

    u = y[:, 0:1]
    v = y[:, 1:2]

    # In 'jacobian', i is the output component and j is the input component
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    v_t = dde.grad.jacobian(y, x, i=1, j=1)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

    f_u = v_t - 0.5*u_xx - (u**2 + v**2)*u
    f_v = u_t + 0.5*v_xx + (u**2 + v**2)*v

    return [f_u, f_v]

# Periodic Boundary conditions
bc_u_0 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0
)
bc_u_1 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0
)

bc_v_0 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1
)
bc_v_1 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1
)

def init_cond_u(x):
    # 2 sech(x)
    return 2 / np.cosh(x[:, 0:1])

def init_cond_v(x):
    return 0

ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, init_cond_v, lambda _, on_initial: on_initial, component=1)

#############################################################################################

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
    num_domain=20000,
    num_boundary=50,
    num_initial=200,
    train_distribution="pseudo",
)

# Network architecture
net = dde.nn.FNN([2] + [100] * 4 + [2], "tanh", "Glorot normal")

dde.optimizers.config.set_LBFGS_options(
    maxcor=50,
    ftol=1.0 * np.finfo(float).eps,
    gtol=1e-08,
    maxiter=10000,
    maxfun=10000,
    maxls=50,
)

# Reload saved model
model_path = "trained_model-16017.ckpt"

loaded_model = dde.Model(data, net)
loaded_model.compile("L-BFGS")
loaded_model.restore(model_path)

print("Model restored successfully.")

# Define specific time steps you want to visualize
t_specific_values = [0.59, 0.79, 0.98]  # Modify these as needed

# Generate x values for plotting
x_plot = np.linspace(x_domain[0], x_domain[1], 100).reshape(-1, 1).astype(np.float32)

# Set up the figure and subplots
fig, axes = plt.subplots(1, len(t_specific_values), figsize=(15, 5))

for i, t_specific in enumerate(t_specific_values):
    # Generate t values (same length as x values) for a specific time step
    t_plot = np.full_like(x_plot, t_specific)

    # Combine x and t into input for model prediction
    X_plot = np.hstack((x_plot, t_plot))
    
    # Predict u and v using the trained model
    prediction = loaded_model.predict(X_plot)
    u_plot, v_plot = prediction[:, 0], prediction[:, 1]

    # Calculate sqrt(u^2 + v^2) for the current time step
    h_plot = np.sqrt(u_plot**2 + v_plot**2)

    # Plot |h(x, t)| for the current time step
    axes[i].plot(x_plot.flatten(), h_plot, color='red', linewidth=2, label=f't = {t_specific}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('|h(x, t)|')
    axes[i].set_title(f'Solution at t = {t_specific}')
    axes[i].legend()
    # axes[i].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Generate the heatmap for sqrt(u**2 + v**2)
xx_plot = np.linspace(x_domain[0], x_domain[1], 100).astype(np.float32)  # 1D array
tt_plot = np.linspace(t_domain[0], t_domain[1], 100).astype(np.float32)  # 1D array

# Create a mesh grid for x and t
xx_mesh, tt_mesh = np.meshgrid(xx_plot, tt_plot)

# Prepare input for model prediction
XX_plot = np.hstack((xx_mesh.flatten()[:, None], tt_mesh.flatten()[:, None]))
uu_plot = loaded_model.predict(XX_plot)

# Reshape the output to match the grid shape for plotting
uu_plot_reshaped = uu_plot.reshape(xx_plot.shape[0], tt_plot.shape[0], 2)  # Ensure correct shape for u and v

# Calculate sqrt(u**2 + v**2) from the reshaped predictions
u_plot = uu_plot_reshaped[:, :, 0]  # Extract u from predictions
v_plot = uu_plot_reshaped[:, :, 1]  # Extract v from predictions
h_plot_reshaped = np.sqrt(u_plot**2 + v_plot**2)  # Calculate sqrt(u^2 + v^2)

# Plotting the heatmap
plt.figure(figsize=(9, 2))
plt.contourf(tt_plot, xx_plot, h_plot_reshaped.T, 100, cmap='YlGnBu')  # Transpose the output for correct orientation
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('|h(x,t)|')
plt.show()

# Generate x values for plotting
x_vals = np.linspace(x_domain[0], x_domain[1], 100).reshape(-1, 1).astype(np.float32)
t_vals = np.linspace(t_domain[0], t_domain[1], 200)  # Time steps for animation

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color='blue', lw=2, label='|h(x, t)|')

ax.set_xlim(x_domain[0], x_domain[1])
ax.set_ylim(0, 5)  # Adjust this based on your data's amplitude
ax.set_xlabel('x')
ax.set_ylabel('|h(x, t)|')
ax.set_title('Wave Function Propagation')
ax.legend()

# Add time annotation
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Initialization function to clear the plot
def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

# Update function for each frame in the animation
def update(frame):
    t_val = t_vals[frame]  # Current time step
    t_input = np.full_like(x_vals, t_val)

    # Combine x and t into input for model prediction
    X_input = np.hstack((x_vals, t_input))

    # Predict u and v using the trained model
    prediction = loaded_model.predict(X_input)
    u_vals, v_vals = prediction[:, 0], prediction[:, 1]

    # Calculate |h(x, t)|
    h_vals = np.sqrt(u_vals**2 + v_vals**2)

    # Update the line data
    line.set_data(x_vals.flatten(), h_vals)

    # Update the time text
    time_text.set_text(f't = {t_val:.3f}')

    return line, time_text

# Create the animation
anim = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, blit=True, interval=50)

# Display the animation
plt.show()

# # Save the animation as MP4 (requires ffmpeg installed)
# anim.save('wave_function_propagation.mp4', writer='ffmpeg', fps=30)

# Or save as a GIF (requires ImageMagick installed)
anim.save('wave_function_propagation.gif', writer='imagemagick', fps=30)

print("Animation saved successfully.")