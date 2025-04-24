import random
import pandas as pd
import datetime

# Function to simulate IoT data
def get_iot_data():
    """Simulate IoT sensor data for PPG and ABP"""
    # In a real app, this would connect to your IoT device
    ppg = random.uniform(60, 100) + random.uniform(-5, 10)  # Simulate PPG data
    abp = random.uniform(120, 140) + random.uniform(-10, 10)  # Simulate ABP data
    return ppg, abp

def create_vital_signs_record(ppg, abp, age, gender, height, weight, bmi, prediction_class, confidence):
    """Create a dictionary with vital signs data for storage"""
    now = datetime.datetime.now()
    
    vital_signs_data = {
        'timestamp': now,
        'ppg': ppg,
        'abp': abp,
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'prediction': prediction_class,
        'confidence': confidence
    }
    
    return vital_signs_data

def add_to_dataframe(df, now, ppg, abp, age, gender, height, weight, bmi, prediction_class, confidence):
    """Add a new row to the dataframe"""
    new_row = pd.DataFrame({
        'Timestamp': [now],
        'PPG': [ppg],
        'ABP': [abp],
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'BMI': [bmi],
        'Prediction': [prediction_class],
        'Confidence': [confidence]
    })
    
    # If df is None, initialize it
    if df is None:
        return new_row
    
    # Otherwise concatenate the new row
    return pd.concat([df, new_row], ignore_index=True)