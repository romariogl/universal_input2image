import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple

class AudioTransformer:
    def transform(self, input_data, save_path: Optional[str] = None, image_size: Tuple[int, int] = (224, 224)):
        """
        Transform audio data into a spectrogram image.
        
        Args:
            input_data: Path to audio file or numpy array
            save_path: Optional path to save the spectrogram image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            image_size: Tuple of (height, width) for the output image size. Default is (224, 224)
            
        Returns:
            np.ndarray: Spectrogram image array
        """
        print("\nStarting audio transformation...")
        
        # Read audio if file path is provided
        if isinstance(input_data, str) and os.path.isfile(input_data):
            print(f"Reading audio from file: {input_data}")
            y, sr = librosa.load(input_data)
        else:
            print("Using provided audio array")
            y = input_data
            sr = 22050  # Default sample rate
            
        print(f"Audio length: {len(y)} samples")
        
        # Compute spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Convert to RGB
        spectrogram = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.tight_layout()
        
        # Get the figure and convert to numpy array
        fig = plt.gcf()
        fig.canvas.draw()
        spectrogram = np.array(fig.canvas.renderer._renderer)
        plt.close()
        
        # Resize to specified dimensions
        spectrogram = cv2.resize(spectrogram, (image_size[1], image_size[0]))
        
        # Normalize to [0, 1] range
        spectrogram = spectrogram.astype(np.float32) / 255.0
        
        # Save image if path is provided
        if save_path:
            print(f"Saving spectrogram to: {save_path}")
            plt.imsave(save_path, spectrogram)
            
        return spectrogram 