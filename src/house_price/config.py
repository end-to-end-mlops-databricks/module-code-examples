from pydantic import BaseModel
from typing import List, Dict, Any
import yaml

class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)