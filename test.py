import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained generator model
generator = tf.keras.models.load_model('generator_model')  # Replace 'generator_model' with the actual path to your trained model


def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = (img / 127.5) - 1.0

    return img

def predict_single_image(input_image_path, generator_model):
    input_image = load_and_preprocess_image(input_image_path)
    input_image = tf.expand_dims(input_image, axis=0)  # Add a batch dimension

    # Generate the clear image from the hazy input image
    clear_image = generator_model(input_image)

    # Denormalize the output image
    clear_image = (clear_image[0] + 1) / 2.0

    return clear_image

# Specify the path to the input hazy image you want to predict
input_image_path = '001.png'

# Predict the clear image
predicted_clear_image = predict_single_image(input_image_path, generator)

# Display the input hazy image and the predicted clear image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(plt.imread(input_image_path))
axes[0].set_title('Input Hazy Image')
axes[0].axis('off')
axes[1].imshow(predicted_clear_image)
axes[1].set_title('Predicted Clear Image')
axes[1].axis('off')
plt.show()




