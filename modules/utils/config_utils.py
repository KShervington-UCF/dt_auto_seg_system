import os
import yaml
from pathlib import Path

class Config:
    """
    Utility class for loading and accessing configuration settings.
    """
    _instance = None
    
    def __new__(cls):
        """Implement the singleton pattern to ensure one config is loaded."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load the configuration file."""
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / 'config.yaml'
        
        # Load the config file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all directories defined in the config."""
        project_root = Path(__file__).resolve().parents[2]
        
        # Create base directories
        directories = [
            self.config['paths']['data_dir'],
            self.config['paths']['raw_dir'],
            self.config['paths']['processed_dir'],
            self.config['paths']['output_dir']
        ]
        
        # Add module-specific directories
        for module in ['preprocessing', 'geopose', 'classification', 
                       'road_segmentation', 'object_segmentation']:
            if module in self.config['paths']:
                if 'input_dir' in self.config['paths'][module]:
                    directories.append(self.config['paths'][module]['input_dir'])
                if 'output_dir' in self.config['paths'][module]:
                    directories.append(self.config['paths'][module]['output_dir'])
        
        # Create tdml integration directories
        if 'tdml_integration' in self.config['paths']:
            if 'output_dir' in self.config['paths']['tdml_integration']:
                directories.append(self.config['paths']['tdml_integration']['output_dir'])
        
        # Create all directories
        for directory in directories:
            dir_path = project_root / directory
            os.makedirs(dir_path, exist_ok=True)
    
    def get_path(self, *keys):
        """
        Get a path from the configuration and convert it to an absolute path.
        
        Args:
            *keys: A sequence of keys to navigate the nested configuration.
                  For example: get_path('preprocessing', 'output_dir')
        
        Returns:
            str: The absolute path.
        """
        # Start at the paths section
        config_section = self.config['paths']
        
        # Navigate through the keys
        for key in keys:
            if key in config_section:
                config_section = config_section[key]
            else:
                raise KeyError(f"Key '{key}' not found in config path.")
        
        # Convert to absolute path
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / config_section)
    
    def get_parameter(self, *keys):
        """
        Get a parameter from the configuration.
        
        Args:
            *keys: A sequence of keys to navigate the nested configuration.
                  For example: get_parameter('preprocessing', 'time_tolerance_ms')
        
        Returns:
            The parameter value.
        """
        # Start at the parameters section
        config_section = self.config['parameters']
        
        # Navigate through the keys
        for key in keys:
            if key in config_section:
                config_section = config_section[key]
            else:
                raise KeyError(f"Key '{key}' not found in config parameters.")
        
        return config_section

# Create a singleton instance for easy import
config = Config()
