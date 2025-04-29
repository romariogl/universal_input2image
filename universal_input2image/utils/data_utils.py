# Data utilities
def detect_input_type(input_data):
    """
    Very simple detector. Could be made smarter later.
    """
    if isinstance(input_data, str):
        return "text"
    if hasattr(input_data, 'shape') and len(input_data.shape) == 2:
        return "tabular"
    if hasattr(input_data, 'edges'):  # NetworkX graph
        return "graph"
    return "unknown"
