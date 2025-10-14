"""
Main entry point for API Testing Agent
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

from core.engine import CoreEngine, APITestRequest
from utils.logger import setup_logger

# Setup logging
logger = setup_logger('api_testing_agent')


async def main(args):
    """Main execution function"""
    try:
        logger.info("Starting API Testing Agent")

        # Parse arguments
        code_files = args.code_files
        language = args.language
        endpoint_url = args.endpoint_url

        # Validate inputs
        if not code_files:
            logger.error("No code files provided")
            return 1

        # Check files exist
        for file_path in code_files:
            if not Path(file_path).exists():
                logger.error(f"File not found: {file_path}")
                return 1

        # Create request
        request = APITestRequest(
            code_files=code_files,
            language=language,
            endpoint_url=endpoint_url,
            max_tests=args.max_tests,
            include_edge_cases=args.edge_cases,
            test_types=args.test_types.split(',') if args.test_types else None
        )

        # Create engine and process
        engine = CoreEngine()
        result = await engine.process_api(request)

        # Handle result
        if result['status'] == 'success':
            logger.info("Test generation completed successfully")
            logger.info(f"Session ID: {result['session_id']}")
            logger.info(f"Total tests: {result['metrics']['total_tests']}")
            logger.info(f"Pass rate: {result['metrics']['pass_rate']:.2%}")
            logger.info(f"Bugs found: {result['metrics']['bugs_found']}")

            # Save report if output specified
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Report saved to: {output_path}")

            # Print summary
            if not args.quiet:
                print("\n" + "=" * 60)
                print("TEST EXECUTION SUMMARY")
                print("=" * 60)
                print(f"Session ID: {result['session_id']}")
                print(f"Total Tests: {result['metrics']['total_tests']}")
                print(f"Passed: {result['metrics']['passed_tests']}")
                print(f"Failed: {result['metrics']['failed_tests']}")
                print(f"Pass Rate: {result['metrics']['pass_rate']:.2f}%")
                print(f"Bugs Found: {result['metrics']['bugs_found']}")
                print(f"Edge Cases Covered: {result['metrics']['edge_cases_covered']}")
                print("=" * 60)

            return 0
        else:
            logger.error(f"Test generation failed: {result.get('error')}")
            return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='API Testing Agent - Automated API test generation and execution'
    )

    # Required arguments
    parser.add_argument(
        'code_files',
        nargs='+',
        help='Path(s) to API code files'
    )

    parser.add_argument(
        '-l', '--language',
        required=True,
        choices=['csharp', 'python', 'java', 'cpp'],
        help='Programming language of the API code'
    )

    parser.add_argument(
        '-u', '--endpoint-url',
        required=True,
        help='Base URL of the API endpoint'
    )

    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        help='Output file path for the report (default: data/reports/report_<timestamp>.json)'
    )

    parser.add_argument(
        '-m', '--max-tests',
        type=int,
        default=50,
        help='Maximum number of tests to generate (default: 50)'
    )

    parser.add_argument(
        '-t', '--test-types',
        help='Comma-separated list of test types (e.g., happy_path,validation,security)'
    )

    parser.add_argument(
        '--no-edge-cases',
        dest='edge_cases',
        action='store_false',
        help='Disable edge case generation'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress console output'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Update log level
    if args.log_level:
        logger.setLevel(args.log_level)

    # Run main
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)