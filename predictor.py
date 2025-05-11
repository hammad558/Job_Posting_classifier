import joblib
import numpy as np

class JobPostClassifier:
    def __init__(self, model_path):
        """Initialize with path to trained model"""
        self.model = joblib.load(model_path)
        self.classes = ['Real', 'Fake']  # Assuming these are your class labels
    
    def predict(self, text):
        """Make prediction on cleaned text"""
        # Vectorize the text (assuming your model expects a list)
        features = [text]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        # Get predicted class (0 or 1)
        predicted_class = self.model.predict(features)[0]
        
        # Get confidence score (probability of predicted class)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence