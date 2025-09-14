import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()


setuptools.setup(
    name="slakonet",
    version="2025.8.1",
    author="Kamal Choudhary",
    author_email="kchoudh2@jhu.edu",
    description="slakonet",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atomgptlab/slakonet",
    packages=setuptools.find_packages(),
    scripts=[
        "slakonet/predict_slakonet.py",
        "slakonet/train_slakonet.py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
