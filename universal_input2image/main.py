import os
import sys
from typing import Optional
import numpy as np
from universal_input2image.utils.detect_input_type import InputTypeDetector
from universal_input2image.transformers.tabular_transformer import TabularTransformer
from universal_input2image.transformers.text_transformer import TextTransformer
from universal_input2image.transformers.graph_transformer import GraphTransformer
from universal_input2image.transformers.time_series_transformer import TimeSeriesTransformer
from universal_input2image.transformers.image_transformer import ImageTransformer

class UniversalInput2Image:
    def __init__(self):
        self.transformers = {
            'tabular': TabularTransformer(),
            'text': TextTransformer(),
            'graph': GraphTransformer(),
            'time_series': TimeSeriesTransformer(),
            'image': ImageTransformer()
        }
    
    def transform(self, input_data, input_type: Optional[str] = None, save_path: Optional[str] = None) -> np.ndarray:
        """
        Transform input data to image using appropriate transformer.
        
        Args:
            input_data: Input data (file path or data object)
            input_type: Optional input type. If not provided, will be detected automatically.
            save_path: Optional path to save the image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            
        Returns:
            np.ndarray: Transformed image
        """
        # Detect input type if not provided
        if input_type is None:
            if isinstance(input_data, str):
                input_type = InputTypeDetector.detect(input_data)
            else:
                # Try to infer type from data structure
                if hasattr(input_data, 'shape'):  # numpy array or pandas DataFrame
                    if len(input_data.shape) == 1:
                        input_type = 'time_series'
                    elif len(input_data.shape) == 2:
                        input_type = 'tabular'
                    else:
                        input_type = 'image'
                elif hasattr(input_data, 'nodes'):  # networkx Graph
                    input_type = 'graph'
                else:
                    input_type = 'text'  # default to text for string data
            
        # Get appropriate transformer
        transformer = self.transformers.get(input_type)
        if transformer is None:
            raise ValueError(f"Unsupported input type: {input_type}")
            
        # Transform to image
        return transformer.transform(input_data, save_path=save_path)

def main():
    """
    Example usage of UniversalInput2Image.
    """
    # Example files (you should replace these with actual file paths)
    example_files = {
        'tabular': 'examples/data/sample.csv',
        'text': 'examples/data/sample.txt',
        'graph': 'examples/data/sample.json',
        'time_series': 'examples/data/timeseries.csv'
    }
    
    # Create transformer instance
    transformer = UniversalInput2Image()
    
    # Process each example file
    for input_type, filepath in example_files.items():
        try:
            print(f"\nProcessing {input_type} file: {filepath}")
            
            # Transform to image
            image = transformer.transform(filepath)
            
            # Print success message and image shape
            print(f"Successfully transformed {input_type} to image")
            print(f"Image shape: {image.shape}")
            
        except Exception as e:
            print(f"Error processing {input_type}: {str(e)}")

if __name__ == "__main__":
    # Add the parent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
