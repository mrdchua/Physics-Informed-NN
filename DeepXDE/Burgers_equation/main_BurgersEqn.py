import deepxde as dde
from deepxde.backend import tf

import numpy as np
import matplotlib.pyplot as plt

dde.config.set_default_float("float64")

def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y*dy_x - (0.01/np.pi)*dy_xx

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:,0:1]), lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160)

net = dde.nn.FNN([2] + [20]*3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(iterations=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

# Plotting predicted solutions for specific times
t_specific_values = [0.25, 0.5, 0.75]  # Change as needed
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, t_specific in enumerate(t_specific_values):
    t_plot = np.full_like(x_plot, t_specific)

    # Prepare input for model prediction
    X_plot = np.hstack((x_plot, t_plot))
    u_plot = model.predict(X_plot)

    # Plot the solution
    axes[i].plot(x_plot, u_plot, color='red', linewidth=2, label=f'u(x, t={t_specific})')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('u(x, t)')
    axes[i].set_title(f'Solution at t = {t_specific}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Generate the heatmap
xx_plot = np.linspace(-1, 1, 100).astype(np.float32)  # 1D array
tt_plot = np.linspace(0, 0.99, 100).astype(np.float32)  # 1D array

# Create a mesh grid for x and t
xx_mesh, tt_mesh = np.meshgrid(xx_plot, tt_plot)

# Prepare input for model prediction
XX_plot = np.hstack((xx_mesh.flatten()[:, None], tt_mesh.flatten()[:, None]))
uu_plot = model.predict(XX_plot)

# Reshape the output to match the grid shape for plotting
uu_plot_reshaped = uu_plot.reshape(xx_plot.shape[0], tt_plot.shape[0])

# Plotting the heatmap
plt.figure(figsize=(9, 2))
plt.contourf(tt_plot, xx_plot, uu_plot_reshaped.T, 100, cmap='rainbow')  # Transpose the output for correct orientation
plt.colorbar(label='u(x,t)')
plt.xlabel('t')
plt.ylabel('x')
# plt.legend(bbox_to_anchor=(1.07, -0.4), loc='lower right')
plt.title('y(x,t)')
# plt.grid(True)
plt.show()
