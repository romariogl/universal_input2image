import os
import pandas as pd
import numpy as np
from typing import Optional, Union
import json
import networkx as nx
from PIL import Image

class InputTypeDetector:
    @staticmethod
    def detect(filepath: str) -> str:
        """
        Detect the type of input based on file extension and content.
        
        Args:
            filepath: Path to the input file
            
        Returns:
            str: One of 'tabular', 'text', 'graph', 'time_series', 'image'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        # Dictionary mapping extensions to potential types
        extension_types = {
            '.csv': ['tabular', 'time_series'],
            '.xlsx': ['tabular', 'time_series'],
            '.xls': ['tabular', 'time_series'],
            '.txt': ['text', 'time_series'],
            '.json': ['graph', 'tabular'],
            '.gml': ['graph'],
            '.graphml': ['graph'],
            '.png': ['image'],
            '.jpg': ['image'],
            '.jpeg': ['image'],
            '.tiff': ['image'],
            '.bmp': ['image']
        }
        
        # Get potential types based on extension
        potential_types = extension_types.get(ext, [])
        
        if not potential_types:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        # If only one potential type, return it
        if len(potential_types) == 1:
            return potential_types[0]
            
        # For multiple potential types, analyze content
        return InputTypeDetector._analyze_content(filepath, potential_types)
    
    @staticmethod
    def _analyze_content(filepath: str, potential_types: list) -> str:
        """
        Analyze file content to determine the most likely type.
        
        Args:
            filepath: Path to the input file
            potential_types: List of potential types based on extension
            
        Returns:
            str: Most likely type
        """
        try:
            if 'tabular' in potential_types and 'time_series' in potential_types:
                # Try to read as DataFrame
                df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
                
                # Check if it's likely time series (has datetime index or column)
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    return 'time_series'
                return 'tabular'
                
            elif 'graph' in potential_types and 'tabular' in potential_types:
                # Try to read as JSON first
                with open(filepath, 'r') as f:
                    content = json.load(f)
                    
                # Check if it has graph-like structure
                if isinstance(content, dict) and ('nodes' in content or 'edges' in content):
                    return 'graph'
                return 'tabular'
                
            elif 'text' in potential_types and 'time_series' in potential_types:
                # Read first few lines to check format
                with open(filepath, 'r') as f:
                    lines = [f.readline() for _ in range(5)]
                    
                # Check if lines contain numeric data
                numeric_lines = sum(1 for line in lines if any(c.isdigit() for c in line))
                if numeric_lines > 3:  # If most lines contain numbers, likely time series
                    return 'time_series'
                return 'text'
                
        except Exception as e:
            # If analysis fails, return the first potential type
            return potential_types[0]
            
        return potential_types[0]  # Default to first potential type if all checks fail 