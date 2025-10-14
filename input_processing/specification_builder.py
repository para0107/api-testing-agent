"""
Build complete API specification from parsed code
"""

import logging
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)


class SpecificationBuilder:
    """Builds complete API specification from parsed components"""

    def __init__(self):
        self.spec_version = "3.0.0"  # OpenAPI version

    def build(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build complete API specification

        Args:
            parsed_data: Parsed code data

        Returns:
            Complete API specification
        """
        logger.info("Building API specification")

        specification = {'openapi': self.spec_version, 'info': self.build_info(parsed_data),
                         'servers': self.build_servers(parsed_data), 'paths': self.build_paths(parsed_data),
                         'components': self.build_components(parsed_data), 'security': self.build_security(parsed_data),
                         'tags': self.build_tags(parsed_data), 'x-test-metadata': {
                'services': parsed_data.get('services', []),
                'validators': parsed_data.get('validators', []),
                'dependencies': parsed_data.get('dependencies', []),
                'business_logic': self.extract_business_logic(parsed_data)
            }}

        # Add custom extensions for testing

        return specification

    def build_info(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build API info section"""
        return {
            'title': 'API Specification',
            'version': '1.0.0',
            'description': 'Auto-generated API specification for testing'
        }

    def build_servers(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build servers section"""
        return [
            {
                'url': 'http://localhost:8080',
                'description': 'Development server'
            },
            {
                'url': 'https://api.example.com',
                'description': 'Production server'
            }
        ]

    def build_paths(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build paths section from endpoints"""
        paths = {}

        endpoints = parsed_data.get('endpoints', [])

        for endpoint in endpoints:
            path = endpoint.get('path') or endpoint.get('route', '')
            method = endpoint.get('method', 'get').lower()

            if path not in paths:
                paths[path] = {}

            paths[path][method] = self.build_operation(endpoint)

        return paths

    def build_operation(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Build operation object for endpoint"""
        operation = {
            'summary': endpoint.get('name', 'Operation'),
            'description': endpoint.get('description', ''),
            'operationId': endpoint.get('name', 'operation'),
            'parameters': self.build_parameters(endpoint),
            'responses': self.build_responses(endpoint)
        }

        # Add request body if needed
        body_params = [p for p in endpoint.get('parameters', [])
                       if p.get('source') == 'body' or p.get('in') == 'body']
        if body_params:
            operation['requestBody'] = self.build_request_body(body_params)

        # Add security if required
        if endpoint.get('authorization', {}).get('required'):
            operation['security'] = self.build_operation_security(endpoint)

        # Add tags
        if endpoint.get('tags'):
            operation['tags'] = endpoint['tags']

        return operation

    def build_parameters(self, endpoint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build parameters list"""
        parameters = []

        for param in endpoint.get('parameters', []):
            if param.get('source') == 'body' or param.get('in') == 'body':
                continue  # Handle in requestBody

            parameter = {
                'name': param.get('name', ''),
                'in': param.get('in') or param.get('source', 'query'),
                'required': param.get('required', False),
                'description': param.get('description', ''),
                'schema': self.build_schema(param)
            }

            parameters.append(parameter)

        return parameters

    def build_schema(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """Build schema for parameter or model"""
        schema = {
            'type': param.get('type', 'string')
        }

        # Add constraints
        constraints = param.get('constraints', {})
        for key, value in constraints.items():
            if key in ['minLength', 'maxLength', 'minimum', 'maximum',
                       'pattern', 'enum', 'format']:
                schema[key] = value

        # Add default value
        if 'default' in param:
            schema['default'] = param['default']

        return schema

    def build_request_body(self, body_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build request body specification"""
        # Assume first body parameter defines the schema
        if not body_params:
            return {}

        param = body_params[0]

        return {
            'required': param.get('required', False),
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            param.get('name', 'body'): self.build_schema(param)
                        }
                    }
                }
            }
        }

    def build_responses(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Build responses section"""
        responses = {
            '200': {
                'description': 'Successful response',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object'
                        }
                    }
                }
            },
            '400': {
                'description': 'Bad request'
            },
            '401': {
                'description': 'Unauthorized'
            },
            '404': {
                'description': 'Not found'
            },
            '500': {
                'description': 'Internal server error'
            }
        }

        return responses

    def build_components(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build components section with schemas"""
        components = {
            'schemas': {},
            'securitySchemes': {}
        }

        # Build schemas from models
        models = parsed_data.get('models', [])
        for model in models:
            schema_name = model.get('name', 'Model')
            components['schemas'][schema_name] = self.build_model_schema(model)

        # Add security schemes
        components['securitySchemes'] = {
            'bearerAuth': {
                'type': 'http',
                'scheme': 'bearer',
                'bearerFormat': 'JWT'
            },
            'apiKey': {
                'type': 'apiKey',
                'in': 'header',
                'name': 'X-API-Key'
            }
        }

        return components

    def build_model_schema(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Build schema for a model"""
        properties = {}
        required = []

        for field in model.get('fields', []):
            field_name = field.get('name', '')
            properties[field_name] = {
                'type': self.map_type_to_openapi(field.get('type', 'string'))
            }

            if field.get('required', False):
                required.append(field_name)

        schema = {
            'type': 'object',
            'properties': properties
        }

        if required:
            schema['required'] = required

        return schema

    def map_type_to_openapi(self, type_str: str) -> str:
        """Map language-specific types to OpenAPI types"""
        type_mapping = {
            'string': 'string',
            'int': 'integer',
            'integer': 'integer',
            'long': 'integer',
            'float': 'number',
            'double': 'number',
            'bool': 'boolean',
            'boolean': 'boolean',
            'array': 'array',
            'list': 'array',
            'object': 'object',
            'dict': 'object',
            'date': 'string',
            'datetime': 'string',
            'uuid': 'string'
        }

        # Handle generic types
        base_type = type_str.lower().split('<')[0] if '<' in type_str else type_str.lower()

        return type_mapping.get(base_type, 'string')

    def build_security(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build security requirements"""
        # Default security requirement
        return [
            {'bearerAuth': []}
        ]

    def build_tags(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build tags for API grouping"""
        tags = set()

        # Extract tags from endpoints
        for endpoint in parsed_data.get('endpoints', []):
            if endpoint.get('tags'):
                tags.update(endpoint['tags'])
            if endpoint.get('controller'):
                tags.add(endpoint['controller'])

        return [{'name': tag, 'description': f'Operations related to {tag}'}
                for tag in sorted(tags)]

    def extract_business_logic(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business logic patterns"""
        business_logic = {
            'validations': [],
            'exceptions': [],
            'workflows': [],
            'dependencies': []
        }

        # Extract validation logic
        validators = parsed_data.get('validators', [])
        for validator in validators:
            business_logic['validations'].append({
                'type': validator.get('type', 'unknown'),
                'target': validator.get('model') or validator.get('field'),
                'rules': validator.get('rules') or validator.get('validations', [])
            })

        # Extract exception handling
        methods = parsed_data.get('methods', [])
        for method in methods:
            if 'throw' in str(method.get('body', '')):
                business_logic['exceptions'].append({
                    'method': method.get('name'),
                    'exceptions': self.extract_exceptions(method.get('body', ''))
                })

        # Extract service dependencies
        services = parsed_data.get('services', [])
        for service in services:
            business_logic['dependencies'].append({
                'type': service.get('type'),
                'name': service.get('name')
            })

        return business_logic

    def extract_exceptions(self, method_body: str) -> List[str]:
        """Extract exception types from method body"""
        import re

        exceptions = []

        # Pattern for different languages
        patterns = [
            r'throw\s+new\s+(\w+Exception)',  # C#/Java
            r'raise\s+(\w+)',  # Python
            r'throw\s+(\w+)',  # C++/JavaScript
        ]

        for pattern in patterns:
            matches = re.findall(pattern, method_body)
            exceptions.extend(matches)

        return list(set(exceptions))

    def build_operation_security(self, endpoint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build security requirements for operation"""
        auth = endpoint.get('authorization', {})

        if auth.get('type') == 'bearer':
            return [{'bearerAuth': auth.get('scopes', [])}]
        elif auth.get('type') == 'apiKey':
            return [{'apiKey': []}]
        else:
            return [{'bearerAuth': []}]  # Default