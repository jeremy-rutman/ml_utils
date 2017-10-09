__author__ = 'jeremy'
from distutils.core import setup

setup(
    name='ml_utils',
    version='0.1.0',
    author='Jeremy Rutman',
    author_email='jeremy.rutman@gmail.com',
    packages=['ml_utils', 'ml_utils.test'],
    scripts=['bin/random_augment.py','bin/convert_tf.py'],
    url='https://github.com/jeremy-rutman/ml-support-code-in-python',
    license='LICENSE.txt',
    description='Useful machine-learning related stuff.',
    long_description=open('README.md').read(),
    install_requires=[
        "OpenCV >= 2.4.0",
        "numpy >= 0.1.4",
    ],
)