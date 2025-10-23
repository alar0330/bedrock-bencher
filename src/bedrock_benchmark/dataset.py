"""
Dataset loading and validation functionality.
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

from .models import BenchmarkItem


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DatasetLoader:
    """Loads and validates JSONL benchmark datasets."""
    
    def load_dataset(self, file_path: str) -> List[BenchmarkItem]:
        """
        Load a JSONL dataset file and return a list of BenchmarkItem objects.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of BenchmarkItem objects
            
        Raises:
            DatasetValidationError: If the file format is invalid or required fields are missing
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        items = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DatasetValidationError(
                            f"Invalid JSON on line {line_num}: {e}"
                        )
                    
                    if not self.validate_format(data):
                        raise DatasetValidationError(
                            f"Invalid format on line {line_num}: missing required fields 'prompt' or 'expected_response'"
                        )
                    
                    # Generate ID if not provided
                    item_id = data.get('id', str(uuid.uuid4()))
                    
                    # Extract metadata (all fields except required ones)
                    metadata = {k: v for k, v in data.items() 
                              if k not in ['id', 'prompt', 'expected_response']}
                    
                    item = BenchmarkItem(
                        id=item_id,
                        prompt=data['prompt'],
                        expected_response=data['expected_response'],
                        metadata=metadata
                    )
                    items.append(item)
                    
        except Exception as e:
            if isinstance(e, (DatasetValidationError, FileNotFoundError)):
                raise
            raise DatasetValidationError(f"Error reading dataset file: {e}")
        
        if not items:
            raise DatasetValidationError("Dataset file is empty or contains no valid items")
        
        return items
    
    def validate_format(self, data: Dict[str, Any]) -> bool:
        """
        Validate that a data item has the required format.
        
        Args:
            data: Dictionary representing a single dataset item
            
        Returns:
            True if the format is valid, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        # Check required fields
        required_fields = ['prompt', 'expected_response']
        for field in required_fields:
            if field not in data:
                return False
            if not isinstance(data[field], str):
                return False
            if not data[field].strip():  # Empty or whitespace-only strings are invalid
                return False
        
        return True