from setuptools import setup, find_packages

setup(
    name='ScatterOT',
    version='0.0.1',
    license='MIT',
    description='Scatter version of OT',
    author='wzm2256',
    url='https://github.com/wzm2256/ScatterOT',
    packages=find_packages(include=['ScatterOT']),
    install_requires=[
        'torch',
        'torch_scatter',
    ],
)