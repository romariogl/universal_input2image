import os
import numpy as np
import pandas as pd
from universal_input2image import UniversalInput2Image

def main():
    # Create examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    
    # Initialize the transformer
    transformer = UniversalInput2Image()
    
    # Example 1: Tabular data from CSV
    print("\nExample 1: Tabular data from CSV")
    csv_path = "examples/data/sample.csv"
    image = transformer.transform(csv_path, save_path='examples/tabular_heatmap.png')
    print(f"Generated image shape: {image.shape}")
    
    # Example 2: Tabular data from DataFrame
    print("\nExample 2: Tabular data from DataFrame")
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.uniform(0, 1, 100),
        'C': np.random.exponential(1, 100)
    })
    image = transformer.transform(df, save_path='examples/dataframe_heatmap.png')
    print(f"Generated image shape: {image.shape}")
    
    # Example 3: Image data
    print("\nExample 3: Image data")
    image_data = np.random.rand(100, 100, 3)  # Random RGB image
    image = transformer.transform(image_data, save_path='examples/random_image.png')
    print(f"Generated image shape: {image.shape}")
    
    # Example 4: Time series data
    print("\nExample 4: Time series data")
    time_series = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    image = transformer.transform(time_series, save_path='examples/time_series.png')
    print(f"Generated image shape: {image.shape}")
    
    print("\nAll examples completed successfully!")
    print("Generated images saved in the 'examples' directory.")

if __name__ == "__main__":
    main() 