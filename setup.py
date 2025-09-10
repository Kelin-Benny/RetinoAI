from setuptools import setup, find_packages

setup(
    name="retinoai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.24.0",
        "Pillow>=9.4.0",
        "Flask>=2.2.0",
        "opencv-python>=4.7.0",
        "scikit-learn>=1.2.0",
        "reportlab>=3.6.0",
    ],
)