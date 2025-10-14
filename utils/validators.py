"""
Validation Utilities Module

This module provides comprehensive validation utilities for API specifications,
test cases, emails, URLs, JSON schemas, and other data structures used in the
API testing agent system.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import ipaddress


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class APISpecValidator:
    """
    Validator for OpenAPI/Swagger specifications

    Validates:
    - Required fields in API spec
    - Endpoint definitions
    - Parameter definitions
    - Schema definitions
    - Response definitions
    """

    REQUIRED_SPEC_FIELDS = ['openapi', 'info', 'paths']
    REQUIRED_INFO_FIELDS = ['title', 'version']
    REQUIRED_OPERATION_FIELDS = ['responses']

    @staticmethod
    def validate_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate OpenAPI specification structure

        Args:
            spec: OpenAPI specification dictionary

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required top-level fields
        for field in APISpecValidator.REQUIRED_SPEC_FIELDS:
            if field not in spec:
                errors.append(f"Missing required field: '{field}'")

        # Validate info section
        if 'info' in spec:
            info_errors = APISpecValidator._validate_info(spec['info'])
            errors.extend(info_errors)

        # Validate paths
        if 'paths' in spec:
            path_errors = APISpecValidator._validate_paths(spec['paths'])
            errors.extend(path_errors)

        # Validate components/schemas if present
        if 'components' in spec and 'schemas' in spec['components']:
            schema_errors = APISpecValidator._validate_schemas(spec['components']['schemas'])
            errors.extend(schema_errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_info(info: Dict[str, Any]) -> List[str]:
        """Validate info section"""
        errors = []

        for field in APISpecValidator.REQUIRED_INFO_FIELDS:
            if field not in info:
                errors.append(f"Missing required info field: '{field}'")

        return errors

    @staticmethod
    def _validate_paths(paths: Dict[str, Any]) -> List[str]:
        """Validate paths section"""
        errors = []

        if not paths:
            errors.append("Paths section is empty")
            return errors

        for path, path_item in paths.items():
            if not path.startswith('/'):
                errors.append(f"Path '{path}' must start with '/'")

            # Validate operations
            valid_methods = ['get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace']
            for method in valid_methods:
                if method in path_item:
                    operation_errors = APISpecValidator._validate_operation(
                        path, method, path_item[method]
                    )
                    errors.extend(operation_errors)

        return errors

    @staticmethod
    def _validate_operation(path: str, method: str, operation: Dict[str, Any]) -> List[str]:
        """Validate operation definition"""
        errors = []

        # Check required fields
        if 'responses' not in operation:
            errors.append(f"Missing 'responses' in {method.upper()} {path}")

        # Validate parameters if present
        if 'parameters' in operation:
            for idx, param in enumerate(operation['parameters']):
                if 'name' not in param:
                    errors.append(f"Parameter {idx} missing 'name' in {method.upper()} {path}")
                if 'in' not in param:
                    errors.append(f"Parameter {idx} missing 'in' in {method.upper()} {path}")
                elif param['in'] not in ['query', 'header', 'path', 'cookie']:
                    errors.append(f"Invalid parameter location '{param['in']}' in {method.upper()} {path}")

        # Validate request body if present
        if 'requestBody' in operation:
            if 'content' not in operation['requestBody']:
                errors.append(f"Missing 'content' in requestBody for {method.upper()} {path}")

        return errors

    @staticmethod
    def _validate_schemas(schemas: Dict[str, Any]) -> List[str]:
        """Validate schema definitions"""
        errors = []

        for schema_name, schema in schemas.items():
            if 'type' not in schema and '$ref' not in schema:
                errors.append(f"Schema '{schema_name}' missing 'type' or '$ref'")

            # Validate properties for object types
            if schema.get('type') == 'object' and 'properties' not in schema:
                errors.append(f"Object schema '{schema_name}' missing 'properties'")

        return errors

    @staticmethod
    def validate_endpoint(endpoint: str) -> bool:
        """
        Validate endpoint format

        Args:
            endpoint: Endpoint path

        Returns:
            True if valid, False otherwise
        """
        if not endpoint:
            return False

        if not endpoint.startswith('/'):
            return False

        # Check for valid characters (alphanumeric, -, _, /, {, })
        pattern = r'^/[\w\-/{}]*$'
        return bool(re.match(pattern, endpoint))

    @staticmethod
    def validate_http_method(method: str) -> bool:
        """
        Validate HTTP method

        Args:
            method: HTTP method

        Returns:
            True if valid, False otherwise
        """
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD', 'TRACE']
        return method.upper() in valid_methods


class TestCaseValidator:
    """
    Validator for test case structures

    Validates test case definitions including assertions, parameters, and test data
    """

    REQUIRED_TEST_FIELDS = ['test_id', 'test_type', 'endpoint', 'method']
    VALID_TEST_TYPES = [
        'happy_path', 'boundary', 'edge_case', 'negative',
        'authentication', 'authorization', 'validation', 'error_handling'
    ]
    VALID_ASSERTION_TYPES = [
        'status_code', 'response_time', 'json_schema', 'field_value',
        'field_exists', 'field_type', 'array_length', 'response_contains'
    ]

    @staticmethod
    def validate_test_case(test_case: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate test case structure

        Args:
            test_case: Test case dictionary

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required fields
        for field in TestCaseValidator.REQUIRED_TEST_FIELDS:
            if field not in test_case:
                errors.append(f"Missing required field: '{field}'")

        # Validate test type
        if 'test_type' in test_case:
            if test_case['test_type'] not in TestCaseValidator.VALID_TEST_TYPES:
                errors.append(f"Invalid test_type: '{test_case['test_type']}'")

        # Validate HTTP method
        if 'method' in test_case:
            if not APISpecValidator.validate_http_method(test_case['method']):
                errors.append(f"Invalid HTTP method: '{test_case['method']}'")

        # Validate endpoint
        if 'endpoint' in test_case:
            if not APISpecValidator.validate_endpoint(test_case['endpoint']):
                errors.append(f"Invalid endpoint format: '{test_case['endpoint']}'")

        # Validate assertions
        if 'assertions' in test_case:
            assertion_errors = TestCaseValidator._validate_assertions(test_case['assertions'])
            errors.extend(assertion_errors)

        # Validate parameters
        if 'parameters' in test_case:
            param_errors = TestCaseValidator._validate_parameters(test_case['parameters'])
            errors.extend(param_errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_assertions(assertions: List[Dict[str, Any]]) -> List[str]:
        """Validate test assertions"""
        errors = []

        if not assertions:
            errors.append("Test case has no assertions")
            return errors

        for idx, assertion in enumerate(assertions):
            if 'type' not in assertion:
                errors.append(f"Assertion {idx} missing 'type'")
            elif assertion['type'] not in TestCaseValidator.VALID_ASSERTION_TYPES:
                errors.append(f"Invalid assertion type: '{assertion['type']}'")

            # Validate specific assertion fields
            if assertion.get('type') == 'status_code' and 'expected' not in assertion:
                errors.append(f"Assertion {idx} of type 'status_code' missing 'expected' field")

            if assertion.get('type') == 'field_value':
                if 'field' not in assertion:
                    errors.append(f"Assertion {idx} of type 'field_value' missing 'field'")
                if 'expected' not in assertion:
                    errors.append(f"Assertion {idx} of type 'field_value' missing 'expected'")

        return errors

    @staticmethod
    def _validate_parameters(parameters: Dict[str, Any]) -> List[str]:
        """Validate test parameters"""
        errors = []

        valid_param_locations = ['query', 'path', 'header', 'body']

        for location, params in parameters.items():
            if location not in valid_param_locations:
                errors.append(f"Invalid parameter location: '{location}'")

            if not isinstance(params, dict):
                errors.append(f"Parameters for '{location}' must be a dictionary")

        return errors


class EmailValidator:
    """Validator for email addresses"""

    # RFC 5322 compliant email regex (simplified)
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        """
        if not email or not isinstance(email, str):
            return False

        # Basic format check
        if not re.match(EmailValidator.EMAIL_PATTERN, email):
            return False

        # Additional checks
        if email.count('@') != 1:
            return False

        local, domain = email.split('@')

        # Check local part
        if len(local) > 64 or len(local) == 0:
            return False

        # Check domain part
        if len(domain) > 255 or len(domain) == 0:
            return False

        # Check for consecutive dots
        if '..' in email:
            return False

        return True


class URLValidator:
    """Validator for URLs"""

    @staticmethod
    def validate_url(url: str, require_scheme: bool = True) -> bool:
        """
        Validate URL format

        Args:
            url: URL to validate
            require_scheme: Whether to require http/https scheme

        Returns:
            True if valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False

        try:
            result = urlparse(url)

            # Check scheme
            if require_scheme:
                if result.scheme not in ['http', 'https']:
                    return False

            # Check netloc (domain)
            if not result.netloc:
                return False

            return True
        except Exception:
            return False

    @staticmethod
    def validate_base_url(url: str) -> bool:
        """
        Validate base URL (must have scheme and domain, no path required)

        Args:
            url: Base URL to validate

        Returns:
            True if valid, False otherwise
        """
        if not URLValidator.validate_url(url):
            return False

        try:
            result = urlparse(url)
            # Base URL should have scheme and netloc
            return bool(result.scheme and result.netloc)
        except Exception:
            return False


class JSONSchemaValidator:
    """Validator for JSON schemas and data"""

    @staticmethod
    def validate_json_string(json_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON string

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            json.loads(json_str)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)

    @staticmethod
    def validate_json_schema(schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate JSON schema structure

        Args:
            schema: JSON schema dictionary

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for type field
        if 'type' not in schema:
            errors.append("Schema missing 'type' field")

        # Validate type value
        valid_types = ['string', 'number', 'integer', 'boolean', 'array', 'object', 'null']
        if 'type' in schema and schema['type'] not in valid_types:
            errors.append(f"Invalid schema type: '{schema['type']}'")

        # Validate object schemas
        if schema.get('type') == 'object':
            if 'properties' not in schema:
                errors.append("Object schema missing 'properties'")

        # Validate array schemas
        if schema.get('type') == 'array':
            if 'items' not in schema:
                errors.append("Array schema missing 'items'")

        return len(errors) == 0, errors

    @staticmethod
    def validate_against_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against JSON schema (basic validation)

        Args:
            data: Data to validate
            schema: JSON schema

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        schema_type = schema.get('type')

        # Type validation
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }

        if schema_type in type_map:
            expected_type = type_map[schema_type]
            if not isinstance(data, expected_type):
                errors.append(f"Expected type '{schema_type}', got '{type(data).__name__}'")

        # Object validation
        if schema_type == 'object' and isinstance(data, dict):
            properties = schema.get('properties', {})
            required = schema.get('required', [])

            # Check required fields
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: '{field}'")

            # Validate properties
            for field, value in data.items():
                if field in properties:
                    field_schema = properties[field]
                    field_valid, field_errors = JSONSchemaValidator.validate_against_schema(
                        value, field_schema
                    )
                    if not field_valid:
                        errors.extend([f"Field '{field}': {err}" for err in field_errors])

        # Array validation
        if schema_type == 'array' and isinstance(data, list):
            items_schema = schema.get('items', {})
            for idx, item in enumerate(data):
                item_valid, item_errors = JSONSchemaValidator.validate_against_schema(
                    item, items_schema
                )
                if not item_valid:
                    errors.extend([f"Item {idx}: {err}" for err in item_errors])

        return len(errors) == 0, errors


class DataValidator:
    """General data validation utilities"""

    @staticmethod
    def validate_port(port: Any) -> bool:
        """Validate port number"""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address (IPv4 or IPv6)"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_timeout(timeout: Any) -> bool:
        """Validate timeout value"""
        try:
            timeout_val = float(timeout)
            return timeout_val > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_status_code(code: Any) -> bool:
        """Validate HTTP status code"""
        try:
            code_num = int(code)
            return 100 <= code_num <= 599
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_non_empty_string(value: Any, field_name: str = "field") -> Tuple[bool, Optional[str]]:
        """Validate that a value is a non-empty string"""
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        if not value.strip():
            return False, f"{field_name} cannot be empty"
        return True, None

    @staticmethod
    def validate_dict(value: Any, field_name: str = "field") -> Tuple[bool, Optional[str]]:
        """Validate that a value is a dictionary"""
        if not isinstance(value, dict):
            return False, f"{field_name} must be a dictionary"
        return True, None

    @staticmethod
    def validate_list(value: Any, field_name: str = "field", min_length: int = 0) -> Tuple[bool, Optional[str]]:
        """Validate that a value is a list"""
        if not isinstance(value, list):
            return False, f"{field_name} must be a list"
        if len(value) < min_length:
            return False, f"{field_name} must have at least {min_length} items"
        return True, None


# Convenience functions
def is_valid_api_spec(spec: Dict[str, Any]) -> bool:
    """Check if API spec is valid"""
    valid, _ = APISpecValidator.validate_spec(spec)
    return valid


def is_valid_test_case(test_case: Dict[str, Any]) -> bool:
    """Check if test case is valid"""
    valid, _ = TestCaseValidator.validate_test_case(test_case)
    return valid


def is_valid_email(email: str) -> bool:
    """Check if email is valid"""
    return EmailValidator.validate_email(email)


def is_valid_url(url: str) -> bool:
    """Check if URL is valid"""
    return URLValidator.validate_url(url)


def is_valid_json(json_str: str) -> bool:
    """Check if JSON string is valid"""
    valid, _ = JSONSchemaValidator.validate_json_string(json_str)
    return valid