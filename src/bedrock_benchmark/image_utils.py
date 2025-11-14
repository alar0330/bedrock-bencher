"""
Image processing utilities for multi-modal embeddings.
"""

import base64
from pathlib import Path
from typing import Set


# Supported image formats
SUPPORTED_FORMATS: Set[str] = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}


def load_and_encode_image(image_path: str) -> str:
    """
    Load image from file path and encode as base64.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64-encoded image string
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
    """
    # Convert to Path object for easier handling
    path = Path(image_path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"Path is not a file: {image_path}"
        )
    
    # Validate file format
    file_extension = path.suffix.lower()
    if file_extension not in SUPPORTED_FORMATS:
        supported_list = ', '.join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported image format: {file_extension}. "
            f"Supported formats: {supported_list}"
        )
    
    # Read and encode the image
    try:
        with open(path, 'rb') as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            return encoded_image
    except IOError as e:
        raise IOError(f"Failed to read image file {image_path}: {str(e)}") from e
