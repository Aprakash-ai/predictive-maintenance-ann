import joblib
import pandas as pd
from tensorflow.keras.models import load_model


# ==========================================================
# Load Saved Model and Scaler
# ==========================================================

def load_prediction_assets():
    """
    Load the trained ANN model and fitted scaler.
    """

    model = load_model("saved_models/predictive_maintenance_ann.keras")
    scaler = joblib.load("saved_models/scaler.pkl")

    return model, scaler


# ==========================================================
# Create Sample Machine Data
# ==========================================================

def create_sample_data():
    """
    Take machine details from the user.
    """

    print("\n" + "=" * 70)
    print("ENTER MACHINE DETAILS")
    print("=" * 70)

    air_temp = float(input("Air Temperature (K): "))
    process_temp = float(input("Process Temperature (K): "))
    rpm = int(input("Rotational Speed (rpm): "))
    torque = float(input("Torque (Nm): "))
    tool_wear = int(input("Tool Wear (min): "))

    machine_type = input("Machine Type (L/M/H): ").upper()

    # One-Hot Encoding for Machine Type

    type_l = 0
    type_m = 0
    type_h = 0

    if machine_type == "L":
        type_l = 1

    elif machine_type == "M":
        type_m = 1

    elif machine_type == "H":
        type_h = 1

    else:
        print("\nInvalid Machine Type!")
        return None

    sample_data = pd.DataFrame({

        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],

        "Type_L": [type_l],
        "Type_M": [type_m],
        "Type_H": [type_h]

    })

    return sample_data


# ==========================================================
# Predict Machine Failure
# ==========================================================

def predict_failure():

    # Load Model & Scaler
    model, scaler = load_prediction_assets()

    print("\n" + "=" * 70)
    print("MODEL AND SCALER LOADED SUCCESSFULLY")
    print("=" * 70)

    # Sample Input
    sample_data = create_sample_data()

    if sample_data is None:
        return

    print("\n" + "=" * 70)
    print("NEW MACHINE DATA")
    print("=" * 70)

    print(sample_data)

    # Numerical Columns
    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    # Scale Features
    scaled_sample = sample_data.copy()

    scaled_sample[numerical_columns] = scaler.transform(
        scaled_sample[numerical_columns]
    )

    print("\n" + "=" * 70)
    print("SCALED MACHINE DATA")
    print("=" * 70)

    print(scaled_sample)

    # Prediction
    prediction = model.predict(scaled_sample)

    print("\n" + "=" * 70)
    print("FAILURE PROBABILITY")
    print("=" * 70)

    print(f"Failure Probability : {prediction[0][0]:.4f}")

    # Final Decision
    if prediction[0][0] >= 0.5:
        print("\nPrediction : MACHINE FAILURE")
    else:
        print("\nPrediction : NO MACHINE FAILURE")