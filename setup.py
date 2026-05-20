from setuptools import find_packages, setup

setup(
    name="primetrade_sentiment_analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    python_requires=">=3.10",
)
