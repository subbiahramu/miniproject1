import tensorflow as tf

# Load the model you trained
model = tf.keras.models.load_model("plant_disease_model.keras")

# View output layer information
model.summary()
