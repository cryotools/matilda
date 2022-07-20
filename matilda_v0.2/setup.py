import setuptools

with open("README.md", "r") as fh:
	description = fh.read()

setuptools.setup(
	name="matilda",
	version="0.2",
	author="Phillip Schuster & Ana-Lena Tappe",
	author_email="phillip.schuster@posteo.de",
	packages=["matilda"],
	description="A package to model water resources in glacierized catchments",
	long_description="Tool for modeling water resources in glacierized catchments. Combines a temperature-index melt model with the conceptual catchment model HBV and a parameterized glacier area/volume re-scaling routine.",
	long_description_content_type="text/markdown",
	url="https://github.com/cryotools/matilda/matilda_v0.2/matilda",
	license='MIT',
	python_requires='>=3.6',
	install_requires=['pandas',
					  'xarray',
					  'matplotlib',
					  'numpy',
					  'scipy',
					  'datetime',
					  'hydroeval',
					  'HydroErr']
)
