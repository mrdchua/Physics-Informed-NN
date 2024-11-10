import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the Neural Network
@tf.keras.utils.register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(PINN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense3 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense4 = tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense5 = tf.keras.layers.Dense(2, activation='linear') # [Re(psi), Im(psi)]
        
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
loaded_model = tf.keras.models.load_model('wave_packet.keras', custom_objects={'PINN': PINN})
print("Model loaded successfully")

loaded_model.summary()

# plotting
domain_x_train = (-10.0, 10.0)
domain_t_train = (0.0, 10)

#######################################################################################################

# Define specific time values for plotting
t_specific_values = [0.0, 1.5, 7.5]  # Change these values as desired

# Generate x values for plotting
x_plot = np.linspace(domain_x_train[0], domain_x_train[1], 100).reshape(-1, 1).astype(np.float32)

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
    u = u_plot[:,0:1]
    v = u_plot[:,1:2]
    u_plot_norm = tf.sqrt(tf.square(u) + tf.square(v))

    # Plot u(x, t_specific) versus x
    axes[i].plot(x_plot, u_plot_norm, color='red', linewidth=3, label=f'|u(x, t={t_specific})|')
    axes[i].plot(x_plot, u, color='blue', linewidth=3, label=f'Re[u(x, t={t_specific}])')
    axes[i].plot(x_plot, v, color='orange', linewidth=3, label=f'Im[u(x, t={t_specific}])')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('|u(x, t)|')
    axes[i].set_title(f'|u(x, t)| at t = {t_specific}')
    axes[i].legend()
#     axes[i].grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()

#######################################################################################################


#######################################################################################################

# Generate x values for plotting
x_vals = np.linspace(domain_x_train[0], domain_x_train[1], 100).reshape(-1, 1).astype(np.float32)
t_vals = np.linspace(domain_t_train[0], domain_t_train[1], 200)  # Time steps for animation

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color='blue', lw=2, label='|u(x, t)|')

ax.set_xlim(domain_x_train[0], domain_x_train[1])
ax.set_ylim(0, 1.2)  # Adjust this based on your data's amplitude
ax.set_xlabel('x')
ax.set_ylabel('|u(x, t)|')
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

    # Predict u and v using the trained model
    prediction = loaded_model(tf.concat([x_vals, t_input], axis=1)).numpy()
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