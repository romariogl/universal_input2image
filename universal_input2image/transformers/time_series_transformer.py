# Time Series Transformer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..base_transformer import BaseTransformer
from typing import Optional

class TimeSeriesTransformer(BaseTransformer):
    def transform(self, input_data, save_path: Optional[str] = None):
        """
        Transform time series data into an image representation.
        
        Args:
            input_data: Path to time series file or pandas DataFrame with datetime index
            save_path: Optional path to save the image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            
        Returns:
            np.ndarray: Image representation of the time series
        """
        # Read data if file path is provided
        if isinstance(input_data, str):
            if input_data.endswith('.csv'):
                df = pd.read_csv(input_data, parse_dates=True, index_col=0)
            else:
                raise ValueError("Unsupported file format. Use CSV files with datetime index.")
        else:
            df = pd.DataFrame(input_data)
            
        # Convert to numpy array
        data = df.values.flatten()
        
        # Normalize to [0, 1] range
        data = (data - data.min()) / (data.max() - data.min())
        
        # Create 2D representation (time vs. amplitude)
        n_points = len(data)
        image = np.zeros((100, n_points))  # Fixed height of 100 pixels
        
        # For each time point, create a vertical line representing the amplitude
        for i, value in enumerate(data):
            height = int(value * 99)  # Scale to 0-99 range
            image[:height+1, i] = 1.0  # Fill from bottom to amplitude
            
        # Convert to RGB
        image_rgb = np.stack([image] * 3, axis=-1)
        
        # Save image if path is provided
        if save_path:
            print(f"Saving image to: {save_path}")
            plt.imsave(save_path, image_rgb)
            
        return image_rgb

    def _extract_numerical_data(self, image: np.ndarray) -> np.ndarray:
        """
        Extract numerical data from the image.
        For time series data, we use the original values.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            return np.mean(image, axis=2)
        return image
