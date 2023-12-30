import os
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

app = Flask(__name__)

# Load the pre-trained generator model
generator = tf.keras.models.load_model('generator_model')  # Replace with the actual model path

# Specify the path for file uploads and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = (img / 127.5) - 1.0
    return img

def predict_single_image(input_image_path, generator_model):
    input_image = load_and_preprocess_image(input_image_path)
    input_image = tf.expand_dims(input_image, axis=0)  # Add a batch dimension
    clear_image = generator_model(input_image)
    clear_image = (clear_image[0] + 1) * 127.5

    return clear_image

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_image_data = None
    ssim_value = None  # Initialize SSIM value

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('Home.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('Home.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_image_path)

            # Predict the clear image
            predicted_clear_image = predict_single_image(input_image_path, generator)

            # Calculate SSIM
            input_image = load_and_preprocess_image(input_image_path)
            input_image = tf.expand_dims(input_image, axis=0)
            ssim_value = tf.image.ssim(input_image, predicted_clear_image, max_val=255).numpy()
            print(f"SSIM: {ssim_value}")

            # Save the PIL Image to a bytes buffer
            img_bytes = io.BytesIO()
            predicted_clear_image_pil = Image.fromarray(predicted_clear_image.numpy().astype('uint8'))
            predicted_clear_image_pil.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # Convert the image to base64 to display in HTML
            predicted_image_data = base64.b64encode(img_bytes.read()).decode('utf-8')

    return render_template('Home.html', predicted_image=predicted_image_data, ssim_value=ssim_value)

if __name__ == '__main__':
    app.run(debug=True)