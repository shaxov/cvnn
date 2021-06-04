import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cvnn",
    version="0.0.3",
    author="Maksym Shpakovych",
    author_email="maksym.shpakovych@gmail.com",
    description="Complex-valued blocks for neural network building.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shaxov/cvnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
