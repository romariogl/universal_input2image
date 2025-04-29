import numpy as np
from scipy import stats
from scipy.stats import yeojohnson, boxcox
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union

def test_normality(data: np.ndarray) -> Tuple[bool, float]:
    """
    Perform normality tests on the data.
    
    Args:
        data: Input data array
        
    Returns:
        Tuple of (is_normal, p_value)
    """
    # Flatten the data for testing
    flat_data = data.flatten()
    
    # Perform Shapiro-Wilk test
    stat, p_value = stats.shapiro(flat_data)
    
    # Consider data normal if p-value > 0.05
    is_normal = p_value > 0.05
    
    return is_normal, p_value

def apply_transformation(data: np.ndarray, method: str = 'yeo-johnson') -> Tuple[np.ndarray, Optional[float]]:
    """
    Apply power transformation to make data more normal.
    
    Args:
        data: Input data array
        method: 'yeo-johnson' or 'box-cox'
        
    Returns:
        Tuple of (transformed_data, lambda)
    """
    flat_data = data.flatten()
    
    if method == 'yeo-johnson':
        transformer = PowerTransformer(method='yeo-johnson')
        transformed_data = transformer.fit_transform(flat_data.reshape(-1, 1))
        return transformed_data.reshape(data.shape), transformer.lambdas_[0]
    else:  # box-cox
        transformed_data, lambda_ = boxcox(flat_data)
        return transformed_data.reshape(data.shape), lambda_

def test_structure_preservation(original: np.ndarray, transformed: np.ndarray) -> bool:
    """
    Test if the structure of the data is preserved after transformation.
    
    Args:
        original: Original data array
        transformed: Transformed data array
        
    Returns:
        Boolean indicating if structure is preserved
    """
    # Calculate correlation between original and transformed data
    corr = np.corrcoef(original.flatten(), transformed.flatten())[0, 1]
    
    # Consider structure preserved if correlation > 0.8
    return corr > 0.8

def visualize_distribution(data: np.ndarray, title: str = "Distribution") -> None:
    """
    Visualize the distribution of the data.
    
    Args:
        data: Input data array
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=50, density=True, alpha=0.6)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show() 