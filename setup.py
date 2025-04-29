# Setup file

from setuptools import setup, find_packages

setup(
    name="universal_input2image",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "networkx",
        "Pillow"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for transforming various input types to images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/universal_input2image",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
