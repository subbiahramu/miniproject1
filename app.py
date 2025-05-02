from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Updated label_map (extend this to match your actual model classes)
# label_map = {
#     0: ("Healthy", "No action needed!"),
#     1: ("Leaf Spot", "Remove affected leaves and avoid overhead watering."),
#     2: ("Blight", "Use copper-based fungicide and remove infected plants."),
#     3: ("Powdery Mildew", "Apply sulfur-based spray and improve air circulation."),
#     # Add more if needed...
# }
# label_map = {
#     0: ("Apple Scab", "Remove infected leaves and apply fungicide."),
#     1: ("Apple Black Rot", "Prune and destroy infected fruits."),
#     2: ("Apple Cedar Rust", "Use resistant varieties and fungicide."),
#     3: ("Apple Healthy", "No action needed."),
#     4: ("Blueberry Healthy", "No action needed."),
#     5: ("Cherry Powdery Mildew", "Apply sulfur-based fungicide."),
#     6: ("Cherry Healthy", "No action needed."),
#     7: ("Corn Gray Leaf Spot", "Rotate crops and use fungicide."),
#     8: ("Corn Common Rust", "Use resistant hybrids and apply fungicide."),
#     9: ("Corn Northern Leaf Blight", "Remove infected leaves."),
#     10: ("Corn Healthy", "No action needed."),
#     11: ("Grape Black Rot", "Prune infected vines and apply fungicide."),
#     12: ("Grape Esca", "Remove dead wood and apply protective sprays."),
#     13: ("Grape Leaf Blight", "Apply copper-based sprays."),
#     14: ("Grape Healthy", "No action needed."),
# }
label_map = {
    0: ("Apple Scab", "Remove infected leaves and apply fungicide, such as a protectant fungicide before wet periods and a systemic fungicide after infection. Ensure good air circulation through pruning."),
    1: ("Apple Black Rot", "Prune and destroy infected fruits and cankers. Maintain good orchard sanitation by removing fallen leaves and mummified fruits. Apply fungicide preventatively, especially during bloom and early fruit development."),
    2: ("Apple Cedar Rust", "Use resistant apple varieties if available. Remove nearby Eastern red cedar trees (the alternate host) if feasible. Apply fungicide protectively before spore release from cedar galls in spring."),
    3: ("Apple Healthy", "No action needed. Continue regular monitoring for any signs of disease or pests."),
    4: ("Blueberry Healthy", "No action needed. Ensure proper soil drainage and pH for optimal growth."),
    5: ("Cherry Powdery Mildew", "Apply sulfur-based or other appropriate fungicides at the first sign of infection. Ensure good air circulation through pruning. Consider using resistant varieties."),
    6: ("Cherry Healthy", "No action needed. Monitor for pests and diseases regularly."),
    7: ("Corn Gray Leaf Spot", "Rotate crops with non-grass species to reduce inoculum. Use tillage to bury infected residue. Apply fungicide if the disease reaches economic thresholds, especially during favorable weather conditions."),
    8: ("Corn Common Rust", "Use resistant hybrid corn varieties. Apply fungicide preventatively or at the first sign of significant infection, especially during cool, humid weather."),
    9: ("Corn Northern Leaf Blight", "Remove heavily infected lower leaves to reduce inoculum. Practice crop rotation and use tillage to bury infected residue. Consider fungicide application on susceptible hybrids under favorable disease conditions."),
    10: ("Corn Healthy", "No action needed. Ensure adequate fertility and water management."),
    11: ("Grape Black Rot", "Prune and destroy infected vines, canes, and mummified berries. Maintain good vineyard sanitation by removing fallen leaves and debris. Apply fungicide preventatively, starting before bloom and continuing through early fruit development."),
    12: ("Grape Esca", "Remove and burn dead wood and infected arms/trunks during dormant pruning. Apply wound protectants after pruning. Ensure proper vineyard management to reduce stress on vines."),
    13: ("Grape Leaf Blight", "Apply copper-based sprays or other recommended fungicides preventatively, especially during wet periods. Ensure good canopy management through pruning and trellising to improve air circulation."),
    14: ("Grape Healthy", "No action needed. Monitor regularly for any signs of disease or pests and maintain good vineyard practices."),
}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded", 400

        image_file = request.files['image']
        if image_file.filename == '':
            return "No selected file", 400

        try:
            # Preprocess image
            image = Image.open(image_file).convert('RGB')
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # Predict
            prediction = model.predict(image)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            print("Prediction Raw:", prediction)
            print("Predicted Class Index:", predicted_class)

            # Get label and suggestion
            class_name, suggestion = label_map.get(
                predicted_class, 
                ("Unknown", "This disease is not recognized. Try another image.")
            )

            return render_template(
                'index.html',
                prediction=class_name,
                suggestion=suggestion,
                confidence=round(confidence, 2)
            )

        except Exception as e:
            return f"Error during prediction: {e}", 500

    # For GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
