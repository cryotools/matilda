import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="matilda",
    version="1.0",
    author="Phillip Schuster, Ana-Lena Tappe & Alexander Georgi",
    author_email="phillip.schuster@geo.hu-berlin.de",
    packages=["matilda"],
    description="A package to model water resources in glacierized catchments",
    long_description="Tool for modeling water resources in glacierized catchments. Combines a temperature-index melt model with the conceptual catchment model HBV and a parameterized glacier area/volume re-scaling routine.",
    long_description_content_type="text/markdown",
    url="https://github.com/cryotools/matilda",
    license='MIT',
    python_requires='>=3.11',
    install_requires=[
	'HydroErr==1.24',
	'hydroeval==0.1.0',
	'matplotlib==3.9.2',
	'numpy==1.26.4',
	'pandas==2.2.3',
	'plotly==5.24.1',
	'scipy==1.10.1',
	'xarray==2024.10.0',
	'DateTime==5.5',
	'pyyaml==6.0.2',
	'spotpy==1.6.2',
	'SciencePlots==2.1.1'
    ],
    extras_require={
    'dev': ['pytest>=7.4.4'],
    },
    include_package_data=True,
    package_data={
        "matilda": ["parameters.json"],
    }
)

