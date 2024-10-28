import deepxde as dde
from deepxde.backend import tf

import numpy as np
import matplotlib.pyplot as plt

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

model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss="MSE")
model.train(iterations=16000, display_every=1000)

dde.optimizers.config.set_LBFGS_options(
    maxcor=50,
    ftol=1.0 * np.finfo(float).eps,
    gtol=1e-08,
    maxiter=50000,
    maxfun=50000,
    maxls=50,
)

model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Save the trained model
model_path = "trained_model"
model.save(model_path)

print(f"Model saved at: {model_path}")