import deepxde as dde
from deepxde.backend import tf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain settings: Infinite square well from x=-10 to x=10
x_domain = [-10.0, 10.0]
t_domain = [0.0, 2.0]  # Time interval

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

    f_u = v_t - 0.5*u_xx
    f_v = u_t + 0.5*v_xx

    return [f_u, f_v]

def init_cond_u(x):
    # u = Re(ψ) part of the initial wave packet
    k = 0.0 # initial momentum / wave number
    sigma = 0.1 # width of Gaussian wave packet
    g = np.sqrt(1.0 / (np.sqrt(np.pi) * sigma) ) * np.exp(-x[:, 0:1]**2 / (2.0*sigma**2))
    return np.cos(k * x[:, 0:1]) * g

def init_cond_v(x):
    k = 0.0 # initial momentum / wave number
    sigma = 0.1 # width of Gaussian wave packet
    g = np.sqrt(1.0 / (np.sqrt(np.pi) * sigma) ) * np.exp(-x[:, 0:1]**2 / (2.0*sigma**2))
    return np.sin(k * x[:, 0:1]) * g

# Dirichlet boundary conditions: ψ(0, t) = ψ(L, t) = 0
def boundary(_, on_boundary):
    return on_boundary

bc_u = dde.icbc.DirichletBC(geomtime, lambda _: 0, boundary, component=0)  # u part
bc_v = dde.icbc.DirichletBC(geomtime, lambda _: 0, boundary, component=1)  # v part

# Initial conditions
ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, init_cond_v, lambda _, on_initial: on_initial, component=1)

#############################################################################################

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_u, bc_v, ic_u, ic_v],
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
model_path = "trained_model-10018.ckpt"

loaded_model = dde.Model(data, net)
loaded_model.compile("L-BFGS")
loaded_model.restore(model_path)

print("Model restored successfully.")

#############################################################################################

# Generate x values for plotting
x_vals = np.linspace(x_domain[0], x_domain[1], 100).reshape(-1, 1).astype(np.float32)
t_vals = np.linspace(t_domain[0], t_domain[1], 200)  # Time steps for animation

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color='blue', lw=2, label='|ψ(x, t)|')

ax.set_xlim(x_domain[0], x_domain[1])
ax.set_ylim(0, 1.2)  # Adjust this based on your data's amplitude
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x, t)|')
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
anim = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, blit=True, interval=300)

# Display the animation
plt.show()

# # Save the animation as MP4 (requires ffmpeg installed)
# anim.save('wave_function_propagation.mp4', writer='ffmpeg', fps=30)

# Or save as a GIF (requires ImageMagick installed)
anim.save('wave_packet_propagation.gif', writer='imagemagick', fps=30)

print("Animation saved successfully.")