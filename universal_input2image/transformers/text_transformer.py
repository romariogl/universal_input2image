# Text Transformer

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from ..base_transformer import BaseTransformer
from typing import Optional, Tuple
import os

class TextTransformer(BaseTransformer):
    def transform(self, input_data, save_path: Optional[str] = None, image_size: Tuple[int, int] = (224, 224)):
        """
        Transform text data into an image representation.
        
        Args:
            input_data: String or path to text file
            save_path: Optional path to save the transformed image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            image_size: Tuple of (height, width) for the output image size. Default is (224, 224)
            
        Returns:
            np.ndarray: Transformed image array
        """
        print("\nStarting text data transformation...")
        
        # Read text if file path is provided
        if isinstance(input_data, str) and os.path.isfile(input_data):
            print(f"Reading text from file: {input_data}")
            with open(input_data, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print("Using provided text")
            text = str(input_data)
            
        print(f"Text length: {len(text)} characters")
        
        # Create word cloud
        wordcloud = WordCloud(
            width=image_size[1],
            height=image_size[0],
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        )
        
        # Generate word cloud
        wordcloud.generate(text)
        
        # Convert to numpy array
        image = wordcloud.to_array()
        
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
        For text data, we use word lengths as numerical representation.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            return np.mean(image, axis=2)
        return image
