__author__ = 'jeremy'
'''
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
'''

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))



setup(
    name='ml_utils',
    version='0.1.0',
    author='Jeremy Rutman',
    author_email='jeremy.rutman@gmail.com',
#    packages=['ml_utils', 'ml_utils.test'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
 #   scripts=['bin/random_augment.py','bin/convert_tf.py'],
    url='https://github.com/jeremy-rutman/ml-support-code-in-python',
    license='LICENSE.txt',
    description='Useful machine-learning related stuff.',
    long_description=open('README.md').read(),
    install_requires=[
    #    "opencv-python >= 2.4.0",
        "numpy >= 0.1.4",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
        ],
    keywords='neural networks support code',
)
