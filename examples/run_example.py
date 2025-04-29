import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from universal_input2image import UniversalInput2Image

def main():
    # Get the absolute path to the examples directory
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(examples_dir, 'data')
    
    # Example files
    example_files = {
        'tabular': os.path.join(data_dir, 'sample.csv'),
        'text': os.path.join(data_dir, 'sample.txt'),
        'graph': os.path.join(data_dir, 'sample.json'),
        'time_series': os.path.join(data_dir, 'timeseries.csv')
    }
    
    # Create transformer instance
    transformer = UniversalInput2Image()
    
    # Process each example file
    for input_type, filepath in example_files.items():
        try:
            print(f"\nProcessing {input_type} file: {filepath}")
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
                
            # Transform to image
            image = transformer.transform(filepath)
            
            # Print success message and image shape
            print(f"Successfully transformed {input_type} to image")
            print(f"Image shape: {image.shape}")
            
        except Exception as e:
            print(f"Error processing {input_type}: {str(e)}")

if __name__ == "__main__":
    main() 