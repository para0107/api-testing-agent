import json
from pathlib import Path

# Load training data
with open('../data/training/qase_training_data.json', 'r') as f:
    data = json.load(f)

# View execution results
if 'execution_results' in data:
    results = data['execution_results']
    passed = sum(1 for r in results if r.get('passed'))
    failed = len(results) - passed

    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # Show first few failures
    print("\nFirst 5 Failed Tests:")
    for i, result in enumerate(results[:5]):
        if not result.get('passed'):
            print(f"  - {result.get('name')}: {result.get('error')}")