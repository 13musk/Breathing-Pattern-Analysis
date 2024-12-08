import numpy as np
import librosa
import sounddevice as sd

class AudioProcessor:
    def __init__(self, sample_rate=22050, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration
        
    def record_audio(self):
        """Record audio for the specified duration"""
        print("Recording...")
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        return audio.flatten()
    
    def extract_features(self, audio):
        """Extract relevant features from audio signal"""
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13
        )
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=audio)
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return {
            'mfccs': mfccs,
            'rms': rms,
            'zcr': zcr
        } 