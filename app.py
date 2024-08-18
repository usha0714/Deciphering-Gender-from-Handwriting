from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained model
best_model_path = 'best_model)vgg16.h5'
best_model = load_model(best_model_path)

# Helper function to process image
def predict_image(model, img_path, index_to_class):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class_name = index_to_class[predicted_index]
    confidence = prediction[0][predicted_index]
    return predicted_class_name, confidence

# Home route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    confidence = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        # Save file to uploads folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        index_to_class={0:'female',1:'male'}
        # Make prediction
        predicted_class_name, conf = predict_image(best_model, img_path, index_to_class)
        prediction = predicted_class_name
        confidence = f"{conf * 100:.2f}%"

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
