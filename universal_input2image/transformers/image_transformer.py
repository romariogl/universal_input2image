# Image Transformer

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..base_transformer import BaseTransformer
from typing import Optional, Tuple
import os

class ImageTransformer(BaseTransformer):
    def transform(self, input_data, save_path: Optional[str] = None, image_size: Tuple[int, int] = (224, 224)):
        """
        Transform image data into a standardized format.
        
        Args:
            input_data: Path to image file or numpy array
            save_path: Optional path to save the transformed image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            image_size: Tuple of (height, width) for the output image size. Default is (224, 224)
            
        Returns:
            np.ndarray: Transformed image array
        """
        print("\nStarting image transformation...")
        
        # Read image if file path is provided
        if isinstance(input_data, str) and os.path.isfile(input_data):
            print(f"Reading image from file: {input_data}")
            image = cv2.imread(input_data)
            if image is None:
                raise ValueError(f"Could not read image from {input_data}")
        else:
            print("Using provided image array")
            image = input_data
            
        print(f"Original image shape: {image.shape}")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
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
        For image data, we use the pixel values directly.
        """
        return image 