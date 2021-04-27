from setuptools import setup, find_packages
import sys



setup(name='icp_error',
      version='0.0.1',
      description='Measurement of displacements across active faults',
      author='Andy Howell',
      author_email='a.howell@gns.cri.nz',
      packages=find_packages(),
      install_requires=["shapely>=1.7.1"]
      )
