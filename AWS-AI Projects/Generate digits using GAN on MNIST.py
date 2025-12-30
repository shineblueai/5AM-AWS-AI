# day_23.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, input_dim=100),
        layers.Reshape((7,7,256)),
        layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
        layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3,3), strides=2, input_shape=(28,28,1), padding='same'),
        layers.LeakyReLU(),
        layers.Conv2D(128, (3,3), strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False

gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train loop (simplified, 100 steps)
noise_dim = 100
batch_size = 32
steps = 100

for step in range(steps):
    # Train Discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    noise = np.random.normal(0,1,(batch_size, noise_dim))
    fake_imgs = generator(noise)
    
    real_labels = np.ones((batch_size,1))
    fake_labels = np.zeros((batch_size,1))
    
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    
    # Train Generator
    noise = np.random.normal(0,1,(batch_size, noise_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

# Generate and show sample
noise = np.random.normal(0,1,(1, noise_dim))
gen_img = generator(noise)
plt.imshow(gen_img[0,:,:,0], cmap='gray')
plt.title('Generated Digit')
plt.show()