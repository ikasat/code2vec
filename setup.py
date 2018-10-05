import ast
import os
import re

from setuptools import setup

PACKAGE_NAME = 'code2vec'

with open(os.path.join(PACKAGE_NAME, '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
    # metadata
    name=PACKAGE_NAME,
    version=version,

    # options
    packages=[PACKAGE_NAME],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.4',
    install_requires=[
        'click==7.0',
        'gensim==3.6.0',
    ],
    extras_require={
        'test': [
            'coverage',
            'pytest>=3',
            'tox',
        ],
        'dev': [
            'flake8',
            'isort',
            'mypy',
            'pyformat',
            'yapf',
        ],
    },
    entry_points='''
        [console_scripts]
        {pkg}={pkg}.cli:main
    '''.format(pkg=PACKAGE_NAME),
)
