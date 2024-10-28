# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:25:58 2024

@author: HP
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import time
import datetime

##############################################################################

# Generate Training Data
batch_size_Nf = 20000
batch_size_Nu = 50
domain_x_train = (-5.0, 5.0)
domain_t_train = (0.0, np.pi/2)

# Collocation points Nf
engine = LatinHypercube(d=2, seed=42)
sample = engine.random(n=batch_size_Nf)

# Scaling factors to keep points within the domain but away from boundaries
scale_factor = 0.98  # For example, using 0.98 to keep it within the domain but not touching the boundary
shift_factor = 0.01  # Shift away from the boundary

x_train_dataset = (sample[:,1] * scale_factor + shift_factor) * (domain_x_train[1]-domain_x_train[0]) + domain_x_train[0]
t_train_dataset = (sample[:,0] * scale_factor + shift_factor) * (domain_t_train[1]-domain_t_train[0]) + domain_t_train[0]

x_train = x_train_dataset.copy().reshape(-1,1).astype(np.float32) # collocation points
t_train = t_train_dataset.copy().reshape(-1,1).astype(np.float32) # collocation points

# # Print domain boundary values
# print(f"Domain x: {domain_x_train}")
# print(f"Domain t: {domain_t_train}")

# # Check if x_train and t_train are within the boundaries
# x_min, x_max = x_train.min(), x_train.max()
# t_min, t_max = t_train.min(), t_train.max()

# print(f"x_train range: {x_min} to {x_max}")
# print(f"t_train range: {t_min} to {t_max}")

# # Verification
# x_within_bounds = (x_min > domain_x_train[0]) and (x_max < domain_x_train[1])
# t_within_bounds = (t_min > domain_t_train[0]) and (t_max < domain_t_train[1])

# print(f"x_train does not touch boundaries: {x_within_bounds}")
# print(f"t_train does not touch boundaries: {t_within_bounds}")

# # boundary data points
# t_train_bounds = np.linspace(domain_t_train[0], domain_t_train[1], batch_size_Nu).reshape(-1,1).astype(np.float32)
# x_train_left = np.full_like(t_train_bounds, domain_x_train[0])
# x_train_right = np.full_like(t_train_bounds, domain_x_train[1])

# # initial data points
# x_train_initial = np.linspace(domain_x_train[0], domain_x_train[1], batch_size_Nu).reshape(-1,1).astype(np.float32)
# t_train_initial = np.full_like(x_train_initial, 0.0)

# u_train_initial = 2.0*(1.0/np.cosh(x_train_initial))

# high resolution data points for accuracy
# Load the .npz file [first option]
data = np.load('schrodinger_data.npz')

# Extract x, t, and u_history
x_sample = data['x']
t_sample = data['t']
u_sample = data['u']

# Randomly select elements from the array
x_train_initial_indices = np.random.choice(len(x_sample), size=batch_size_Nu, replace=False)
x_train_initial_indices = np.sort(x_train_initial_indices)

t_train_bounds_indices = np.random.choice(len(t_sample), size=batch_size_Nu, replace=False)
t_train_bounds_indices = np.sort(t_train_bounds_indices)

# initial data points
x_train_initial = x_sample[x_train_initial_indices].reshape(-1,1).astype(np.float32)
t_train_initial = np.full_like(x_train_initial, 0.0, dtype=np.float32)

u_train_initial_Re = np.real(u_sample[0,x_train_initial_indices]).reshape(-1,1).astype(np.float32)
u_train_initial_Im = np.imag(u_sample[0,x_train_initial_indices]).reshape(-1,1).astype(np.float32)
u_train_initial_norm = np.sqrt( u_train_initial_Re**2 + u_train_initial_Im**2 )

# random points
t_train_bounds_high_res = t_sample[t_train_bounds_indices].reshape(-1,1).astype(np.float32)
x_train_left_high_res = np.full_like(t_train_bounds_high_res, x_sample[0])
x_train_right_high_res = np.full_like(t_train_bounds_high_res, x_sample[-1])

u_train_left_high_res_Re = np.real(u_sample[t_train_bounds_indices,0]).reshape(-1,1).astype(np.float32)
u_train_left_high_res_Im = np.imag(u_sample[t_train_bounds_indices,0]).reshape(-1,1).astype(np.float32)
u_train_right_high_res_Re = np.real(u_sample[t_train_bounds_indices,-1]).reshape(-1,1).astype(np.float32)
u_train_right_high_res_Im = np.imag(u_sample[t_train_bounds_indices,-1]).reshape(-1,1).astype(np.float32)

# plotting collocation and initial points
plt.scatter(x_train, t_train, alpha=1, linewidths=1e-6, label=f'$N_f$ = {batch_size_Nf} collocation points')
plt.scatter(x_train_left_high_res, t_train_bounds_high_res, color='red', alpha=0.6, linewidths=1e-6, label=f'$N_u$ = {batch_size_Nu*3} initial and boundary data')
plt.scatter(x_train_right_high_res, t_train_bounds_high_res, color='red', alpha=0.6, linewidths=1e-6)
plt.scatter(x_train_initial, t_train_initial, color='red', alpha=0.6, linewidths=1e-6)
plt.xlabel('x')
plt.ylabel('t')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# # plotting collocation and initial points
# plt.scatter(x_train, t_train, alpha=1, linewidths=1e-6, label=f'$N_f$ = {batch_size_Nf} collocation points')
# plt.scatter(x_train_left, t_train_bounds, color='red', alpha=0.6, linewidths=1e-6, label=f'$N_u$ = {batch_size_Nu*3} initial and boundary data')
# plt.scatter(x_train_right, t_train_bounds, color='red', alpha=0.6, linewidths=1e-6)
# plt.scatter(x_train_initial, t_train_initial, color='red', alpha=0.6, linewidths=1e-6)
# plt.xlabel('x')
# plt.ylabel('t')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()

##############################################################################

# Define the Neural Network
@tf.keras.utils.register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(PINN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense3 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense4 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense5 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense6 = tf.keras.layers.Dense(2, activation='linear') # u_Re, u_Im
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x
    
    def get_config(self):
        config = super(PINN, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# Define the Physics Loss
def physics_loss(model, x_f, t_f):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_f)
        tape.watch(t_f)
        
        u = model(tf.concat([x_f, t_f], axis=1))
        u_Re, u_Im = u[..., 0], u[..., 1]
        u_norm_sqrd = tf.square(u_Re) + tf.square(u_Im)
        V = -u_norm_sqrd
        u_Re_x = tape.gradient(u_Re, x_f)
        u_Im_x = tape.gradient(u_Im, x_f)
        u_Re_t = tape.gradient(u_Re, t_f)
        u_Im_t = tape.gradient(u_Im, t_f)
        
    u_Re_xx = tape.gradient(u_Re_x, x_f)
    u_Im_xx = tape.gradient(u_Im_x, x_f)
    del tape
    
    # residual loss
    f_Re = -u_Im_t + 0.5*u_Re_xx - V*u_Re
    f_Im = u_Re_t + 0.5*u_Im_xx - V*u_Im
    
    f_norm = tf.sqrt( tf.square(f_Re) + tf.square(f_Im) )
    
    return tf.reduce_mean(tf.square(f_norm))

# def boundary_condition(model, x_f_left, x_f_right, t_f):
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(x_f_left)
#         tape.watch(x_f_right)
        
#         u_left = model(tf.concat([x_f_left, t_f], axis=1))
#         u_left_Re, u_left_Im = u_left[..., 0], u_left[..., 1]
        
#         u_right = model(tf.concat([x_f_right, t_f], axis=1))
#         u_right_Re, u_right_Im = u_right[..., 0], u_right[..., 1]
    
#     u_left_Re_x = tape.gradient(u_left_Re, x_f_left)
#     u_left_Im_x = tape.gradient(u_left_Im, x_f_left)
#     u_right_Re_x = tape.gradient(u_right_Re, x_f_right)
#     u_right_Im_x = tape.gradient(u_right_Im, x_f_right)
#     del tape
    
#     # boundary loss
#     loss_Re = u_left_Re - u_right_Re
#     loss_Im = u_left_Im - u_right_Im
#     loss_x_Re = u_left_Re_x - u_right_Re_x
#     loss_x_Im = u_left_Im_x - u_right_Im_x
    
#     return tf.reduce_mean(tf.square(loss_Re)) + tf.reduce_mean(tf.square(loss_Im)) + \
#         tf.reduce_mean(tf.square(loss_x_Re)) + tf.reduce_mean(tf.square(loss_x_Im))

# def initial_condition(model, x_f, t_f, u_data):
#     u = model(tf.concat([x_f, t_f], axis=1))
#     u_Re, u_Im = u[..., 0], u[..., 1]
    
#     loss_Re = u_Re - u_data
#     loss_Im = u_Im - 0.0
    
#     return tf.reduce_mean(tf.square(loss_Re)) + tf.reduce_mean(tf.square(loss_Im))

def loss_from_high_res_data(model, x_f, t_f, u_data_Re, u_data_Im):
    u = model(tf.concat([x_f, t_f], axis=1))
    u_Re, u_Im = u[..., 0], u[..., 1]
    
    loss_Re = u_Re - u_data_Re
    loss_Im = u_Im - u_data_Im
    
    loss_norm = tf.sqrt( tf.square(loss_Re) + tf.square(loss_Im) )
    
    return tf.reduce_mean(tf.square(loss_norm))

#############################################################################

# Instantiate the Model
pinn = PINN()

# Define the Optimizer
optimizer = tf.keras.optimizers.Adam()

# Training Step Function
@tf.function
def train_step(x_train, t_train, x_train_initial, t_train_initial, u_train_initial_Re, u_train_initial_Im, t_train_bounds_high_res, \
               x_train_left_high_res, u_train_left_high_res_Re, u_train_left_high_res_Im, x_train_right_high_res, u_train_right_high_res_Re, u_train_right_high_res_Im):
    with tf.GradientTape() as tape:
        pde_loss = physics_loss(pinn, x_train, t_train)
        initial_loss = loss_from_high_res_data(pinn, x_train_initial, t_train_initial, u_train_initial_Re, u_train_initial_Im)
        bc_left_loss = loss_from_high_res_data(pinn, x_train_left_high_res, t_train_bounds_high_res, u_train_left_high_res_Re, u_train_left_high_res_Im)
        bc_right_loss = loss_from_high_res_data(pinn, x_train_right_high_res, t_train_bounds_high_res, u_train_right_high_res_Re, u_train_right_high_res_Im)
        loss = pde_loss + initial_loss + bc_left_loss + bc_right_loss
        
    gradients = tape.gradient(loss, pinn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, pinn.trainable_variables))
    return loss

#############################################################################

# List to store the loss values
loss_history = []
epoch_history = [0,]

# Define the loss threshold
loss_threshold = 1e-4

# Early stopping
early_stopping_patience = 500
best_loss = float('inf')
patience_counter = 0

# Start the timer
start_time = time.time()

# Training Loop until loss < loss_threshold
epoch = 0
while True:
    loss = train_step(x_train, t_train, x_train_initial, t_train_initial, u_train_initial_Re, u_train_initial_Im, t_train_bounds_high_res, \
                   x_train_left_high_res, u_train_left_high_res_Re, u_train_left_high_res_Im, x_train_right_high_res, u_train_right_high_res_Re, u_train_right_high_res_Im)
    loss_value = loss.numpy()
    loss_history.append(loss_value)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")
        
        # Save the model in Keras format
        model_save_path_0 = "interrupted_Schrodinger_equation_v7.keras"
        pinn.save(model_save_path_0)
        print(f"Model saved to {model_save_path_0}")
        
    if loss_value < best_loss:
        best_loss = loss_value
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch} as loss did not improve for {early_stopping_patience} epochs.")
        print(f"Training stopped at epoch {epoch} with loss {loss_value}")
        break
    
    if loss_value < loss_threshold:
        print(f"Training stopped at epoch {epoch} as loss reached {loss_value}")
        break
    
    epoch += 1
    epoch_history.append(epoch)

# End the timer
end_time = time.time()

# Calculate total execution time
execution_time = end_time - start_time

# Convert to hours, minutes, seconds
execution_time_formatted = str(datetime.timedelta(seconds=int(execution_time)))

print(f"Total training time: {execution_time_formatted}")
        
# Plot the Loss vs. Epochs
plt.plot(epoch_history, loss_history, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Save the model in Keras format
model_save_path = "Schrodinger_equation_v7.keras"
pinn.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Save original weights
original_weights = pinn.get_weights()

# Load the model
loaded_pinn = tf.keras.models.load_model(model_save_path, custom_objects={'PINN': PINN})

# Check if the weights match
loaded_weights = loaded_pinn.get_weights()

for orig, loaded in zip(original_weights, loaded_weights):
    if not np.array_equal(orig, loaded):
        print("Weights do not match after loading!")
    else:
        print("Weights match.")

# Check prediction consistency
x_test = np.linspace(domain_x_train[0], domain_x_train[1], 100).reshape(-1,1).astype(np.float32)
t_test = np.linspace(domain_t_train[0], domain_t_train[1], 100).reshape(-1,1).astype(np.float32)

x_test_tensor = tf.convert_to_tensor(x_test)
t_test_tensor = tf.convert_to_tensor(t_test)

original_prediction = pinn(tf.concat([x_test_tensor, t_test_tensor], axis=1)).numpy()
loaded_prediction = loaded_pinn(tf.concat([x_test_tensor, t_test_tensor], axis=1)).numpy()

if not np.allclose(original_prediction, loaded_prediction, atol=1e-5):
    print("Predictions do not match after loading!")
else:
    print("Predictions match.")