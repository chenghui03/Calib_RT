import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


setuptools.setup(
    name="pycalib_rt",
    version="0.1.2",
    author="yichi zhang",
    author_email="kszyc1001@163.com",
    description="Calib-RT is designed for RT (retention time) calibration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenghui03/Calib_RT/tree/main/", 
    packages=setuptools.find_packages(),
    package_data={'calib_rt': ['data/*.npz',]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.26.0',
                      'pandas>=2.1.1',
                      'networkx>=3.1',
                      'statsmodels>=0.14.0',
                      'scipy>=1.11.3'],
    python_requires='>=3.10'
)
