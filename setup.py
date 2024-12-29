from setuptools import find_packages
from distutils.core import setup

setup(
    name='genesis_lg',
    version='0.1.0',
    author='Baix3XiaoRuo',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='hanyang_chou@icloud.com',
    description='Forked from Genesis environments for Legged Robots',
    install_requires=['gym',
                      'rsl-rl',
                      'matplotlib',
                      'open3d',
                      
                      ]
)