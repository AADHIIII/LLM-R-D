"""
Test suite for verifying the basic setup and configuration.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, TrainingConfig
from utils.logging import setup_logging, get_logger
from utils.exceptions import LLMOptimizationError, DatasetValidationError
from utils.error_handler import get_error_handler


class TestSetup(unittest.TestCase):
    """Test basic setup and configuration."""
    
    def test_settings_loading(self):
        """Test that settings can be loaded."""
        settings = get_settings()
        self.assertIsNotNone(settings)
        self.assertIsInstance(settings.api_port, int)
        self.assertIsInstance(settings.flask_debug, bool)
    
    def test_training_config(self):
        """Test training configuration creation."""
        config = TrainingConfig(
            base_model="gpt2",
            epochs=3,
            batch_size=4,
        )
        
        self.assertEqual(config.base_model, "gpt2")
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 4)
        
        # Test conversion to dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["base_model"], "gpt2")
    
    def test_logging_setup(self):
        """Test logging configuration."""
        logger = setup_logging("test_logger")
        self.assertIsNotNone(logger)
        
        # Test getting logger
        test_logger = get_logger("test")
        self.assertIsNotNone(test_logger)
    
    def test_custom_exceptions(self):
        """Test custom exception classes."""
        # Test base exception
        base_error = LLMOptimizationError(
            "Test error",
            error_code="TEST_ERROR",
            details={"test": "value"},
            suggested_actions=["Try again"]
        )
        
        error_dict = base_error.to_dict()
        self.assertEqual(error_dict["error_code"], "TEST_ERROR")
        self.assertEqual(error_dict["message"], "Test error")
        self.assertEqual(error_dict["details"]["test"], "value")
        
        # Test specific exception
        dataset_error = DatasetValidationError(
            "Invalid dataset",
            invalid_fields=["field1", "field2"]
        )
        
        self.assertEqual(dataset_error.invalid_fields, ["field1", "field2"])
    
    def test_error_handler(self):
        """Test error handler functionality."""
        error_handler = get_error_handler()
        self.assertIsNotNone(error_handler)
        
        # Test handling a ValueError
        try:
            raise ValueError("Test value error")
        except ValueError as e:
            error_response = error_handler.handle_exception(e)
            self.assertEqual(error_response["error_code"], "INVALID_VALUE")
            self.assertIn("Invalid value", error_response["message"])
    
    def test_directory_structure(self):
        """Test that required directories exist or can be created."""
        settings = get_settings()
        
        # Check if directories exist or can be created
        directories = [
            settings.model_storage_path,
            settings.dataset_storage_path,
            os.path.dirname(settings.log_file),
        ]
        
        for directory in directories:
            if directory:
                # Directory should exist or be creatable
                if not os.path.exists(directory):
                    try:
                        os.makedirs(directory, exist_ok=True)
                        self.assertTrue(os.path.exists(directory))
                    except Exception as e:
                        self.fail(f"Could not create directory {directory}: {e}")


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_config_imports(self):
        """Test config module imports."""
        try:
            from config.settings import Settings, TrainingConfig, get_settings
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Failed to import config modules: {e}")
    
    def test_utils_imports(self):
        """Test utils module imports."""
        try:
            from utils.logging import setup_logging, get_logger
            from utils.exceptions import LLMOptimizationError
            from utils.error_handler import get_error_handler
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Failed to import utils modules: {e}")


if __name__ == "__main__":
    unittest.main()