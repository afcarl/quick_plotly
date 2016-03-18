from setuptools import setup, find_packages 
from codecs import open  
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='stacked_generalization',
    packages=find_packages(),
    version='0.0.0',
    description='Simple interface to Plotly.js for python.',
    long_description=long_description,
    url='https://github.com/dustinstansbury/quick_plotly',
    author='Dustin Stansbury',
    author_email='https://dustin.stansbury@gmail.com',
    keywords='data visualization, plotly, python',

    install_requires=['numpy',
                      'scipy',
                      'seaborn',
                      'plotly'
                      ],
)
