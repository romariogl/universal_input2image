# Tabular Transformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from ..base_transformer import BaseTransformer
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

class TabularTransformer(BaseTransformer):
    def transform(self, input_data, save_path: Optional[str] = None, image_size: Tuple[int, int] = (224, 224)):
        """
        Transform tabular data into an image representation.
        
        Args:
            input_data: pandas DataFrame or path to CSV file
            save_path: Optional path to save the transformed image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            image_size: Tuple of (height, width) for the output image size. Default is (224, 224)
            
        Returns:
            np.ndarray: Transformed image array
        """
        print("\nStarting tabular data transformation...")
        
        # Read data if file path is provided
        if isinstance(input_data, str) and os.path.isfile(input_data):
            print(f"Reading data from file: {input_data}")
            df = pd.read_csv(input_data)
        else:
            print("Using provided DataFrame")
            df = pd.DataFrame(input_data)
            
        print(f"Input data shape: {df.shape}")
        
        # Normalize data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df)
        
        # Create heatmap
        plt.figure(figsize=(10, 10))
        sns.heatmap(normalized_data, cmap='viridis', cbar=True)
        plt.title('Tabular Data Heatmap')
        
        # Convert plot to numpy array
        plt.tight_layout()
        canvas = plt.gca().figure.canvas
        canvas.draw()
        image = np.array(canvas.renderer.buffer_rgba())
        plt.close()
        
        # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        image = cv2.resize(image, (image_size[1], image_size[0]))
        
        # Normalize to [0, 1] range
        image = image.astype(np.float32) / 255.0
        
        # Save image if path is provided
        if save_path:
            print(f"Saving image to: {save_path}")
            plt.imsave(save_path, image)
            
        return image

    def _extract_numerical_data(self, image: np.ndarray) -> np.ndarray:
        """
        Extract numerical data from the image.
        For tabular data, we can use the original data directly.
        """
        return image
