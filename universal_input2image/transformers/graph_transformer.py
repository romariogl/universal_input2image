# Graph Transformer

import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from ..base_transformer import BaseTransformer
from typing import Optional

class GraphTransformer(BaseTransformer):
    def transform(self, input_data, save_path: Optional[str] = None):
        """
        Transform graph data into an image using network visualization.
        
        Args:
            input_data: Path to graph file (JSON, GML, etc.) or networkx Graph
            save_path: Optional path to save the image. If provided, saves the image to file.
                      Should include file extension (e.g., 'output.png', 'output.jpg')
            
        Returns:
            np.ndarray: Image representation of the graph
        """
        # Read graph if file path is provided
        if isinstance(input_data, str):
            if input_data.endswith('.json'):
                with open(input_data, 'r') as f:
                    data = json.load(f)
                G = nx.Graph()
                if 'nodes' in data:
                    G.add_nodes_from(data['nodes'])
                if 'edges' in data:
                    G.add_edges_from(data['edges'])
            elif input_data.endswith('.gml'):
                G = nx.read_gml(input_data)
            else:
                raise ValueError("Unsupported graph file format")
        else:
            G = input_data
            
        # Create figure with two subplots
        fig, (ax1, ax2) = self._create_figure(figsize=(15, 8))
        
        # Subplot 1: Graph Visualization
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color='skyblue', 
                node_size=500, font_size=8, font_weight='bold')
        ax1.set_title('Graph Visualization')
        
        # Subplot 2: Degree Distribution
        degrees = [d for n, d in G.degree()]
        ax2.hist(degrees, bins=20, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Degree Distribution')
        
        # Convert to image
        image = self._save_figure_to_image(fig, save_path)
        
        # Create numerical representation for statistical tests
        numerical_data = np.array(degrees)
        
        # Run statistical tests
        _, is_approved = self._transform_with_tests(numerical_data)
        
        if not is_approved:
            raise ValueError("Failed to create a valid transformation after multiple attempts")
            
        return image

    def _extract_numerical_data(self, image: np.ndarray) -> np.ndarray:
        """
        Extract numerical data from the image.
        For graph data, we use degree distribution as numerical representation.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            return np.mean(image, axis=2)
        return image
