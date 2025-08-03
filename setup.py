import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slakonet",
    version="2025.8.1",
    author="Kamal Choudhary",
    author_email="kchoudh2@jhu.edu",
    description="slakonet",
    slakonet=[
        "numpy>=1.22.0",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atomgptlab/slakonet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
