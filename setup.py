from setuptools import setup

with open("README.md","r") as fh:
    long_description=fh.read()
    

setup(name='pdffit', 
     version=0.1,
     description='Fit a Lognormal + Power law distribution to data',
     py_modules=["LNPL_functions", "LNPLPL_functions", "pdf_fitter"],
     package_dir={'': 'src'},
     classifiers = ["Programming Language :: Python :: 3.8", "License :: OSI Approved :: MIT License"],
     long_description=long_description,
     long_description_content_type="text/markdown",
     install_requires = ["numpy >=1.0", "scipy >=1.0"],
     url = "https://github.com/shivankhullar/PDF_Fit",
     author = "Shivan Khullar",
     author_email = "shivankhullar@gmail.com",
     )