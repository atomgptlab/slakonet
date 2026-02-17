import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()


setuptools.setup(
    name="slakonet",
    version="2025.9.1",
    author="Kamal Choudhary",
    author_email="kchoudh2@jhu.edu",
    description="slakonet",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atomgptlab/slakonet",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "predict_slakonet=slakonet.predict_slakonet:main",
            "train_slakonet=slakonet.train_slakonet:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
