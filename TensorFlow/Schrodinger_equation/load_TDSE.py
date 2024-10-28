import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the Neural Network
@tf.keras.utils.register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(PINN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(80, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(80, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense3 = tf.keras.layers.Dense(80, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense4 = tf.keras.layers.Dense(80, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
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
loaded_model = tf.keras.models.load_model('interrupted_Schrodinger_equation_v7.keras', custom_objects={'PINN': PINN})
print("Model loaded successfully")

loaded_model.summary()

# plotting
domain_x_train = (-5.0, 5.0)
domain_t_train = (0.0, np.pi/2.0)

# Define specific time values for plotting
t_specific_values = [0.0, 0.79, 0.98]  # Change these values as desired

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
    u_plot_Re, u_plot_Im = u_plot[..., 0], u_plot[..., 1]
    u_plot_norm = tf.sqrt(tf.square(u_plot_Re) + tf.square(u_plot_Im))

    # Plot u(x, t_specific) versus x
    axes[i].plot(x_plot, u_plot_norm, color='red', linewidth=3, label=f'|u(x, t={t_specific})|^2')
    # axes[i].plot(x_plot, u_plot_Re, color='blue', linewidth=3, label=f'Re[u(x, t={t_specific}])')
    # axes[i].plot(x_plot, u_plot_Im, color='orange', linewidth=3, label=f'Re[u(x, t={t_specific}])')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('|u(x, t)|^2')
    axes[i].set_title(f'|u(x, t)|^2 at t = {t_specific}')
    axes[i].legend()
#     axes[i].grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()