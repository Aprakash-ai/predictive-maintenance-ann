import joblib
from tensorflow.keras.models import load_model

model = load_model("saved_models/predictive_maintenance_ann.keras")
scaler = joblib.load("saved_models/scaler.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Scaler feature count:", scaler.n_features_in_)