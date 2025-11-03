from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]

setup(
    name="rfae",  # Package name (used in imports)
    version="0.1.0",  # Version number
    packages=find_packages(),  # Automatically finds `rfae/` and submodules
    install_requires=parse_requirements('requirements.txt'),
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
