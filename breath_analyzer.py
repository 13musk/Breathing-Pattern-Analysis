import numpy as np
from tensorflow.keras import Model

class BreathAnalyzer:
    def __init__(self, model: Model):
        self.model = model
        self.breathing_patterns = [
            'normal_breathing',
            'ujjayi_breathing',
            'kapalabhati',
            'deep_breathing'
        ]
    
    def analyze_breath(self, features):
        """Analyze breathing pattern from extracted features"""
        # Prepare features for model input
        mfccs = features['mfccs']
        rms = features['rms']
        zcr = features['zcr']
        
        # Combine features
        combined_features = np.concatenate([
            mfccs.mean(axis=1),
            rms.mean(axis=1),
            zcr.mean(axis=1)
        ])
        
        # Reshape for model input
        model_input = combined_features.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(model_input)
        pattern_idx = np.argmax(prediction)
        
        return {
            'pattern': self.breathing_patterns[pattern_idx],
            'confidence': float(prediction[0][pattern_idx])
        } 