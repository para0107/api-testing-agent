"""
Base parser class for all language parsers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for language-specific parsers"""

    def __init__(self):
        self.parsed_data = {}
        self.endpoints = []
        self.services = []
        self.validators = []

    @abstractmethod
    def parse(self, code_files: List[str]) -> Dict[str, Any]:
        """
        Parse code files and extract API information

        Args:
            code_files: List of file paths to parse

        Returns:
            Dictionary containing parsed API information
        """
        pass

    @abstractmethod
    def extract_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract API endpoints from code

        Args:
            code: Source code string

        Returns:
            List of endpoint definitions
        """
        pass

    @abstractmethod
    def extract_methods(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract methods/functions from code

        Args:
            code: Source code string

        Returns:
            List of method definitions
        """
        pass

    @abstractmethod
    def extract_parameters(self, method_code: str) -> List[Dict[str, Any]]:
        """
        Extract parameters from a method

        Args:
            method_code: Method source code

        Returns:
            List of parameter definitions
        """
        pass

    @abstractmethod
    def extract_validation_rules(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract validation rules from code

        Args:
            code: Source code string

        Returns:
            List of validation rules
        """
        pass

    def read_file(self, file_path: str) -> str:
        """
        Read a source code file

        Args:
            file_path: Path to the file

        Returns:
            File contents as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine parsing results from multiple files

        Args:
            results: List of parsing results

        Returns:
            Combined parsing results
        """
        combined = {
            'endpoints': [],
            'services': [],
            'validators': [],
            'methods': [],
            'models': []
        }

        for result in results:
            for key in combined.keys():
                if key in result:
                    combined[key].extend(result[key])

        return combined

    def extract_dependencies(self, code: str) -> List[str]:
        """
        Extract dependencies/imports from code

        Args:
            code: Source code string

        Returns:
            List of dependencies
        """
        # Default implementation - override in specific parsers
        return []

    def extract_models(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract data models/DTOs from code

        Args:
            code: Source code string

        Returns:
            List of model definitions
        """
        # Default implementation - override in specific parsers
        return []

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a source code file and extract API endpoints

        Args:
            file_path: Path to the source code file

        Returns:
            Dictionary containing parsed data with endpoints and metadata
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Parse the source code
            parsed_data = self.parse(source_code)

            # Add file metadata
            parsed_data['file_path'] = file_path
            parsed_data['language'] = self.language

            logger.info(f"Successfully parsed {file_path}: found {len(parsed_data.get('endpoints', []))} endpoints")

            return parsed_data

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise
