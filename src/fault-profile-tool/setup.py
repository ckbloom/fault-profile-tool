from setuptools import setup, find_packages
import sys

mac_install_requires = ["numpy>=1.17",
                        "matplotlib>=3.1.1",
                        "geopandas>=0.6.1",
                        "netcdf4>=1.4.2",
                        "ipython>=7.9.0",
                        "PyQt5>=5.13.1",
                        "scipy>=1.3.1",
                        "rasterio>=1.1.0"]
linux_install_requires = ["numpy>=1.17",
                          "matplotlib>=3.1.1",
                          "geopandas>=0.6.1",
                          "netcdf4>=1.4.2",
                          "ipython>=7.9.0",
                          "PyQt5>=5.12.1",
                          "scipy>=1.3.1",
                          "rasterio>=1.1.0"]
windows_install_requires = []

install_dic = {"darwin": mac_install_requires, "linux": linux_install_requires, "win32": windows_install_requires}

setup(name='fault_profile_tool',
      version='0.0.2',
      description='Measurements of displacements across active faults',
      author='Andy Howell',
      author_email='a.howell@gns.cri.nz',
      packages=find_packages(),
      install_requires=install_dic[sys.platform]
      )
