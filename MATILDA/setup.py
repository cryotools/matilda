from setuptools import setup, find_packages
setup(name='MATILDA_slim',
version='0.1',
description='Modeling wATer resources In gLacierizeD cAtchments',
url='#',
author='ana',
author_email='testr@gmail.com',
license='MIT',
install_requires=['pandas' , 'xarray', 'matplotlib', 'numpy', 'scipy', 'datetime', 'hydroeval'], 
packages=['MATILDA_slim'],
zip_safe=False)
