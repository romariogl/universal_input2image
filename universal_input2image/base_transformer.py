# Base Transformer
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import matplotlib.pyplot as plt
from .utils.statistical_tests import (
    test_normality,
    apply_transformation,
    test_structure_preservation,
    visualize_distribution
)
import logging

class BaseTransformer(ABC):
    def __init__(self):
        self.max_attempts = 3
        self.current_attempt = 0
        self.transformed_data = None
        self.transformation_history = []
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self._logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    @abstractmethod
    def transform(self, input_data, save_path: Optional[str] = None, image_size: Tuple[int, int] = (224, 224)):
        """
        Transform input data into a standardized image format.
        
        Args:
            input_data: The input data to transform
            save_path: Optional path to save the transformed image
            image_size: Tuple of (height, width) for the output image size. Default is (224, 224)
            
        Returns:
            np.ndarray: The transformed image data
        """
        pass

    def _transform_with_tests(self, numerical_data: np.ndarray, attribute_names: Optional[List[str]] = None) -> Tuple[np.ndarray, bool]:
        """
        Transform data with statistical testing flow.
        
        Args:
            numerical_data: Numerical data to test
            attribute_names: Optional list of attribute names for logging
            
        Returns:
            Tuple of (transformed_data, is_approved)
        """
        print(f"\nStarting statistical tests (attempt {self.current_attempt + 1}/{self.max_attempts})")
        
        # Ensure data is 2D
        if len(numerical_data.shape) == 1:
            numerical_data = numerical_data.reshape(-1, 1)
            
        n_attributes = numerical_data.shape[1]
        if attribute_names is None:
            attribute_names = [f"Attribute_{i+1}" for i in range(n_attributes)]
            
        while self.current_attempt < self.max_attempts:
            transformed_data = numerical_data.copy()
            all_normal = True
            transformation_results = {}
            
            # Test each attribute separately
            for i in range(n_attributes):
                attr_name = attribute_names[i]
                print(f"\nTesting attribute: {attr_name}")
                
                # Get attribute data
                attr_data = numerical_data[:, i]
                
                # Step 1: Test normality
                print("Testing normality...")
                is_normal, p_value = test_normality(attr_data)
                print(f"Normality test result: {'Normal' if is_normal else 'Not Normal'} (p-value: {p_value:.4f})")
                
                if not is_normal:
                    all_normal = False
                    print("Data is not normal, applying transformation...")
                    # Step 2: Apply transformation if not normal
                    transformed_attr, lambda_ = apply_transformation(attr_data)
                    print(f"Applied transformation with lambda: {lambda_:.4f}")
                    
                    # Step 3: Test normality again
                    print("Testing normality after transformation...")
                    is_normal, p_value = test_normality(transformed_attr)
                    print(f"Normality test after transformation: {'Normal' if is_normal else 'Not Normal'} (p-value: {p_value:.4f})")
                    
                    if is_normal:
                        transformed_data[:, i] = transformed_attr
                        transformation_results[attr_name] = {
                            'original_normal': False,
                            'transformed_normal': True,
                            'lambda': lambda_
                        }
                    else:
                        transformation_results[attr_name] = {
                            'original_normal': False,
                            'transformed_normal': False,
                            'lambda': lambda_
                        }
                        self.current_attempt += 1
                        break
                else:
                    print("Data is already normal, no transformation needed")
                    transformation_results[attr_name] = {
                        'original_normal': True,
                        'transformed_normal': True,
                        'lambda': None
                    }
            
            # If any attribute failed transformation, try again
            if not all_normal and any(not result['transformed_normal'] for result in transformation_results.values()):
                print("Some attributes failed transformation, trying again...")
                continue
            
            # Step 4: Test structure preservation for each transformed attribute
            print("\nTesting structure preservation...")
            for i, attr_name in enumerate(attribute_names):
                if transformation_results[attr_name]['original_normal']:
                    continue
                    
                structure_preserved = test_structure_preservation(
                    numerical_data[:, i],
                    transformed_data[:, i]
                )
                print(f"Structure preservation for {attr_name}: {'Preserved' if structure_preserved else 'Not Preserved'}")
                
                if not structure_preserved:
                    print("Structure not preserved, trying again...")
                    self.current_attempt += 1
                    break
            else:  # If no break occurred in the loop
                print("All tests passed successfully!")
                return transformed_data, True
        
        # If we reach here, all attempts failed
        print(f"Failed to create a valid transformation after {self.max_attempts} attempts")
        return numerical_data, False

    def _extract_numerical_data(self, image: np.ndarray) -> np.ndarray:
        """
        Extract numerical data from the image.
        This is a basic implementation that can be overridden by child classes.
        
        Args:
            image: Input image array
            
        Returns:
            Numerical data array
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            return np.mean(image, axis=2)
        return image

    def _create_figure(self, figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a matplotlib figure with specified size.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Tuple of (figure, axes)
        """
        return plt.subplots(figsize=figsize)

    def _save_figure_to_image(self, fig: plt.Figure, save_path: Optional[str] = None) -> np.ndarray:
        """
        Save matplotlib figure to numpy array and optionally to file.
        
        Args:
            fig: Matplotlib figure
            save_path: Optional path to save the image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            
        Returns:
            Image as numpy array
        """
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        
        if save_path:
            print(f"Saving image to: {save_path}")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.close(fig)
        return image
