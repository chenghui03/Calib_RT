import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


setuptools.setup(
    name="calib_rt",                                     
    version="0.0.3",                                        
    author="yichi zhang",                                       
    author_email="kszyc1001@163.com",                   
    description="calib_rt",                            
    long_description=long_description,                      
    long_description_content_type="text/markdown",          
    url="https://gitee.com/chenghui03/calib_-rt/tree/release/",                              
    packages=setuptools.find_packages(),                    
    package_data={'calib_rt': ['data/*.npz',]},
    classifiers=[                                           
        "Programming Language :: Python :: 3",              
        "License :: OSI Approved :: MIT License",           
        "Operating System :: OS Independent",               
    ],
    install_requires=['numpy','pandas','networkx','statsmodels','scipy'],  
    python_requires='>=3'
)