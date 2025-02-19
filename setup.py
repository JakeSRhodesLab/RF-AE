from setuptools import setup, find_packages

setup(
    name="rfae",  # Package name (used in imports)
    version="0.1.0",  # Version number
    packages=find_packages(),  # Automatically finds `rfae/` and submodules
    install_requires=[
        "numpy~=1.26.4",
        "pandas~=2.0.3",
        "scikit-learn~=1.4.0",
        "torch~=2.2.0",
        "scipy~=1.10.1",
        "graphtools~=1.5.3",
        "phate~=1.0.11",
        "joblib~=1.3.2",
        "rootutils~=1.0.7"
    ],
    author="JakeSRhodesLab",
    description = "Random Forest Autoencoders (RF-AE), a neural network-based framework for out-of-sample kernel extension that combines the flexibility of autoencoders with the supervised learning strengths of random forests and the geometry captured by RF-PHATE.", 
    url="https://github.com/JakeSRhodesLab/RF-AE",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version
)
