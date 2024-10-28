# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 03:19:06 2024

@author: HP
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# Load the model
loaded_model = tf.keras.models.load_model('Burgers_equation_v6.keras', custom_objects={'PINN': PINN})
print("Model loaded successfully")

# Define specific time values for plotting
t_specific_values = [0.25, 0.5, 0.75]  # Change these values as desired

# Generate x values for plotting
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)

# Set up the subplot grid
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Loop over each t_specific value to generate and plot the solution
for i, t_specific in enumerate(t_specific_values):
    # Create an array of the same shape as x with the specific time t
    t_plot = np.full_like(x_plot, t_specific)

    # Convert to tensors
    x_plot_tensor = tf.convert_to_tensor(x_plot)
    t_plot_tensor = tf.convert_to_tensor(t_plot)

    # Get predictions
    u_plot = loaded_model(tf.concat([x_plot_tensor, t_plot_tensor], axis=1)).numpy()

    # Plot u(x, t_specific) versus x
    axes[i].plot(x_plot, u_plot, color='red', linewidth=3, label=f'u(x, t={t_specific})')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('u(x, t)')
    axes[i].set_title(f'u(x, t) at t = {t_specific}')
    axes[i].legend()
#     axes[i].grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()

# Generate the heatmap

# Generate Training Data
batch_size_Nf = 10000
batch_size_Nu = 100
domain_x_train = (-1.0, 1.0)
domain_t_train = (0.0, 1.0)

# boundary data points
t_train_bounds = np.linspace(domain_t_train[0], domain_t_train[1], batch_size_Nu).reshape(-1,1).astype(np.float32)
x_train_left = np.full_like(t_train_bounds, -1.0)
x_train_right = np.full_like(t_train_bounds, 1.0)

# initial data points
x_train_initial = np.linspace(domain_x_train[0], domain_x_train[1], batch_size_Nu).reshape(-1,1).astype(np.float32)
t_train_initial = np.full_like(x_train_initial, 0.0)

# Generate x and t values for the grid
x_plot = np.linspace(domain_x_train[0], domain_x_train[1], 100).astype(np.float32)
t_plot = np.linspace(domain_t_train[0], domain_t_train[1], 100).astype(np.float32)

# Create a meshgrid
X, T = np.meshgrid(x_plot, t_plot)

# Flatten the meshgrid and stack to pass through the model
X_flat = X.flatten().reshape(-1, 1)
T_flat = T.flatten().reshape(-1, 1)

# Convert to tensors
X_tensor = tf.convert_to_tensor(X_flat)
T_tensor = tf.convert_to_tensor(T_flat)

# Get predictions for the entire grid
u_flat = loaded_model(tf.concat([X_tensor, T_tensor], axis=1)).numpy()

# Reshape the predictions to match the shape of the grid
U = u_flat.reshape(X.shape)

# Generate the heatmap
plt.figure(figsize=(9, 2))
plt.contourf(T, X, U, 100, cmap='rainbow')
plt.colorbar()
plt.scatter(t_train_bounds, x_train_left, color='black', marker='X', linewidths=0.1, label=f'$N_u$ = {batch_size_Nu*3} initial and boundary data')
plt.scatter(t_train_bounds, x_train_right, color='black', marker='X', linewidths=0.1)
plt.scatter(t_train_initial, x_train_initial, color='black', marker='X', linewidths=0.1)
plt.xlabel('t')
plt.ylabel('x')
plt.legend(bbox_to_anchor=(1.07, -0.4), loc='lower right')
plt.title('u(x,t)')
plt.show()