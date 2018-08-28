from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ml_recsys_tools',
    version='0.5.3',
    description='Tools for recommendation systems development',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DomainGroupOSS/ml-recsys-tools',
    author='Domain group (Arthur Deygin)',
    author_email='arthurdgn@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='recommendations machine learning',
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=['requests', 'numpy', 'scipy', 'pandas',
                      'scikit_learn', 'lightfm', 'implicit', 'psutil',
                      'scikit_optimize', 'gmaps', 'boto3', 'redis',
                      'sklearn_pandas', 'matplotlib', 'flask'],
)

## rm -rf dist && python3 setup.py sdist bdist_wheel && twine upload dist/*
