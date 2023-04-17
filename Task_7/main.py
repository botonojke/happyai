import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(path, height, width):
    img = Image.open(path)
    img = img.resize((height, width), resample=Image.LANCZOS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def save_image(img_array, path):
    img_array = np.clip(img_array, 0, 255).astype("uint8")
    img = Image.fromarray(img_array[0])
    img.save(path)

def deep_dream(base_image, dream_model, steps=100, step_size=0.01):
    # Define the loss function
    def calc_loss(img, model):
        # Create a tensor from the image
        img_batch = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = tf.keras.applications.inception_v3.preprocess_input(img_batch)
        img_batch = tf.convert_to_tensor(img_batch)

        # Calculate the activations of the chosen layers
        layer_activations = model(img_batch)

        # Calculate the loss as the sum of the layer activations
        losses = []
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)
        return tf.reduce_sum(losses)

    # Convert the base image to a tensor
    base_image = tf.keras.preprocessing.image.img_to_array(base_image)

    # Define the optimizer
    opt = tf.optimizers.Adam(learning_rate=0.01, decay=1e-6)

    # Run the optimization loop
    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(base_image)
            loss = calc_loss(base_image, dream_model)

        gradients = tape.gradient(loss, base_image)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        base_image = base_image + gradients * step_size
        base_image = tf.clip_by_value(base_image, -1, 1)

    # Convert the tensor back to an image
    base_image = tf.keras.preprocessing.image.array_to_img(base_image[0])
    return base_image

# Load the InceptionV3 model without the classification layer
dream_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# Load the base image
img_path = "/home/boton/Rep/happyai/Task_7/sky.jpg"
height, width = 600, 600
base_image = load_image(img_path, height, width)

# Generate the deep dream image
dream_image = deep_dream(base_image, dream_model, steps=100, step_size=0.01)

# Save the result
save_image(dream_image, "/home/boton/Rep/happyai/Task_7/sky_output.jpg")
