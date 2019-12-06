import setuptools

description = "Scripts and functions for data extraction for the International Brain Lab"

setuptools.setup(
    name="iblscripts",
    version="0.0.1",
    author="Jai Bhagat",
    author_email="jai.bhagat@internationalbrainlab.org",
    description=description,
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/int-brain-lab/iblscripts",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)