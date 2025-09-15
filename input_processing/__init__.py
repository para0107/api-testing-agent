"""
Input processing module for parsing and analyzing API code
"""

from .parser_factory import ParserFactory
from .endpoint_extractor import EndpointExtractor
from .sepcification_builder import SpecificationBuilder
from .validator_extractor import ValidatorExtractor


class InputProcessor:
    """Main input processor combining all components"""

    def __init__(self):
        self.parser_factory = ParserFactory()
        self.endpoint_extractor = EndpointExtractor()
        self.specification_builder = SpecificationBuilder()
        self.validator_extractor = ValidatorExtractor()

    def parse_code(self, code_files, language):
        """Parse code files based on language"""
        parser = self.parser_factory.get_parser(language)
        return parser.parse(code_files)

    def build_specification(self, parsed_data):
        """Build API specification from parsed data"""
        return self.specification_builder.build(parsed_data)

    def extract_business_logic(self, parsed_data):
        """Extract business logic from parsed data"""
        return self.specification_builder.extract_business_logic(parsed_data)

    def extract_validation_rules(self, parsed_data):
        """Extract validation rules from parsed data"""
        return self.validator_extractor.extract(parsed_data)


__all__ = [
    'InputProcessor',
    'ParserFactory',
    'EndpointExtractor',
    'SpecificationBuilder',
    'ValidatorExtractor'
]