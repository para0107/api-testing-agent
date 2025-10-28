"""
Setup script for API Testing Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
def read_requirements(filename):
    requirements_path = Path(__file__).parent / filename
    if requirements_path.exists():
        with open(requirements_path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = ''
if readme_path.exists():
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='api-testing-agent',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='AI-powered automated API testing with RAG and RL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/api-testing-agent',
    packages=find_packages(exclude=['tests', 'scripts']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'api-test-agent=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
)