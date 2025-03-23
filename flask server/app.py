from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
import io
import cv2

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the trained Keras model
model = tf.keras.models.load_model("best_base_model_new.keras")

# Define class labels (Modify as per your model's classes)
class_labels = ["0", "1", "2", "3", "4" , "5", "6"]  

# Function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((64, 64))  # Resize to match model input size
#     image = np.array(image) / 255.0   # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

target_size = (64,64)

classes={0:('actinic keratoses and intraepithelial carcinomae(Cancer)'),
         1:('basal cell carcinoma(Cancer)'),
         2:('benign keratosis-like lesions(Non-Cancerous)'),
         3:('dermatofibroma(Non-Cancerous)'),
         4:('melanocytic nevi(Non-Cancerous)'),
         5:('pyogenic granulomas and hemorrhage(Can lead to cancer)'),
         6:('melanoma(Cancer)')}

def preprocess_image(img):
    """Resizes an image to (64, 64, 3) while maintaining aspect ratio by padding."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")  # Ensure RGB format
        img = np.array(img)  # Convert PIL Image to NumPy array

    old_size = img.shape[:2]  # (height, width)
    ratio = min(float(target_size[0]) / old_size[0], float(target_size[1]) / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Black padding
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Convert to float32 and normalize
    new_im = new_im.astype("float32") / 255.0  

    return new_im




# API Endpoint to receive image and return prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))  # Read image file
    processed_image = preprocess_image(image)  # Preprocess

    # ✅ Fix: Add batch dimension (1, 64, 64, 3)
    processed_image = np.expand_dims(processed_image, axis=0)

    prediction = model.predict(processed_image)  # Get prediction
    predicted_class = class_labels[np.argmax(prediction)]  # Get highest probability class
    predicted_class = int(predicted_class)
  
    response =  jsonify({"prediction": classes[predicted_class]})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
