# Universal Input2Image Library

A versatile Python library for transforming various types of input data into visual representations. This library provides a unified interface to convert different data formats (tabular data, time series, images, etc.) into meaningful visualizations.

## Features

- Transform tabular data (CSV, DataFrames) into heatmap visualizations
- Convert time series data into line plots
- Process image data with various transformations
- Support for multiple input formats
- Easy-to-use API with consistent interface
- Customizable visualization parameters

## Installation

You can install the library using pip:

```bash
pip install universal-input2image
```

## Requirements

- Python >= 3.6
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- networkx
- Pillow

## Usage

Here's a quick example of how to use the library:

```python
from universal_input2image import UniversalInput2Image

# Initialize the transformer
transformer = UniversalInput2Image()

# Transform tabular data from CSV
image = transformer.transform("path/to/your/data.csv", save_path='output.png')

# Transform DataFrame
import pandas as pd
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
image = transformer.transform(df, save_path='output.png')

# Transform time series data
import numpy as np
time_series = np.sin(np.linspace(0, 10, 100))
image = transformer.transform(time_series, save_path='output.png')
```

## Examples

The repository includes several example scripts demonstrating different use cases:

1. Tabular data transformation
2. DataFrame visualization
3. Image processing
4. Time series visualization

Run the examples using:
```bash
python run_example.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or suggestions, please open an issue in the GitHub repository. 