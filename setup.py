import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_deeco",
    version="0.0.1",
    author='Milad Ashqi Abdullah, Michal Töpfer, Tomáš Bureš, Petr Hnětynka, Martin Kruliš',
    author_email='topfer@d3s.mff.cuni.cz',
    description='ML-DEECo: Machine Learning-enabled Component Model for Dynamically Adapting Systems',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/smartarch/ml_deeco',
    keywords=[],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.21.3",
        "seaborn>=0.11.1",
        "matplotlib>=3.5.1",
    ],
)
