# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:36:32 2024

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
batch_size_Nf = 10000
batch_size_Nu = 100
domain_x_train = (-1.0, 1.0)
domain_t_train = (0.0, 1.0)

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

# boundary data points
t_train_bounds = np.linspace(domain_t_train[0], domain_t_train[1], batch_size_Nu).reshape(-1,1).astype(np.float32)
x_train_left = np.full_like(t_train_bounds, -1.0)
x_train_right = np.full_like(t_train_bounds, 1.0)

u_train_bounds = np.full_like(t_train_bounds, 0.0)

# initial data points
x_train_initial = np.linspace(domain_x_train[0], domain_x_train[1], batch_size_Nu).reshape(-1,1).astype(np.float32)
t_train_initial = np.full_like(x_train_initial, 0.0)

u_train_initial = -np.sin(np.pi*x_train_initial)


# plotting collocation and initial points
plt.scatter(x_train, t_train, alpha=1, linewidths=1e-6, label=f'$N_f$ = {batch_size_Nf} collocation points')
plt.scatter(x_train_left, t_train_bounds, color='red', alpha=0.6, linewidths=1e-6, label=f'$N_u$ = {batch_size_Nu*3} initial and boundary data')
plt.scatter(x_train_right, t_train_bounds, color='red', alpha=0.6, linewidths=1e-6)
plt.scatter(x_train_initial, t_train_initial, color='red', alpha=0.6, linewidths=1e-6)
plt.xlabel('x')
plt.ylabel('t')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

##############################################################################

# Define the Neural Network
@tf.keras.utils.register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(PINN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense3 = tf.keras.layers.Dense(50, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense4 = tf.keras.layers.Dense(50, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense5 = tf.keras.layers.Dense(1, activation='linear')
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
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
        u_x = tape.gradient(u, x_f)
        u_t = tape.gradient(u, t_f)
        
    u_xx = tape.gradient(u_x, x_f)
    del tape
    
    # residual loss
    f = u_t + u*u_x - (0.01/np.pi)*u_xx
    
    return tf.reduce_mean(tf.square(f))

def boundary_condition_loss(model, x_f, t_f, u_data):
    u_pred = model(tf.concat([x_f, t_f], axis=1))
    
    bc_loss = u_pred - u_data
    
    return tf.reduce_mean(tf.square(bc_loss))

#############################################################################

# Instantiate the Model
pinn = PINN()

# Define the Optimizer
optimizer = tf.keras.optimizers.Adam()

# Training Step Function
@tf.function
def train_step(x_train, t_train, x_train_left, x_train_right, t_train_bounds, u_train_bounds, \
               x_train_initial, t_train_initial, u_train_initial):
    with tf.GradientTape() as tape:
        pde_loss = physics_loss(pinn, x_train, t_train)
        bc_loss_left = boundary_condition_loss(pinn, x_train_left, t_train_bounds, u_train_bounds)
        bc_loss_right = boundary_condition_loss(pinn, x_train_right, t_train_bounds, u_train_bounds)
        initial_cond_loss = boundary_condition_loss(pinn, x_train_initial, t_train_initial, u_train_initial)
        loss = pde_loss + bc_loss_left + bc_loss_right + initial_cond_loss
        
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
    loss = train_step(x_train, t_train, x_train_left, x_train_right, t_train_bounds, \
                      u_train_bounds, x_train_initial, t_train_initial, u_train_initial)
    loss_value = loss.numpy()
    loss_history.append(loss_value)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")
        
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
# plt.title('Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Save the model in Keras format
model_save_path = "Burgers_equation_v6.keras"
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