import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model('cnn_model.keras')

# Create a TFLite converter object from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Enable optimizations for smaller size and faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model to disk
with open('cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as cnn_model.tflite")
