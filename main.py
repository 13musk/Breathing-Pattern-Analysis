from src.audio_processor import AudioProcessor
from src.breath_analyzer import BreathAnalyzer
from src.model import create_model
import numpy as np

def main():
    # Initialize components
    audio_processor = AudioProcessor(duration=5)
    
    # Create and load model (in practice, you'd load pre-trained weights)
    model = create_model(input_shape=(15,))  # Adjust shape based on your features
    
    breath_analyzer = BreathAnalyzer(model)
    
    try:
        while True:
            # Record audio
            audio = audio_processor.record_audio()
            
            # Extract features
            features = audio_processor.extract_features(audio)
            
            # Analyze breathing pattern
            result = breath_analyzer.analyze_breath(features)
            
            # Display results
            print(f"\nDetected Breathing Pattern: {result['pattern']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            # Ask user if they want to continue
            if input("\nPress Enter to continue or 'q' to quit: ").lower() == 'q':
                break
                
    except KeyboardInterrupt:
        print("\nStopping the program...")

if __name__ == "__main__":
    main() 