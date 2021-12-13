from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f]
setup(
    name='AI Stock Advisor',
    version='.1',
    description='An intelligent ML system to find undervalued/overvalued stocks.',
    author='Nolan Braman',
    license="GNU3.0",
    long_description=long_description,
    author_email='nolan.braman@gmail.com',
    packages=['AI Stock Advisor'],  # same as name
    python_requires='>=3.6',
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU 3.0 License",
        "Operating System :: OS Independent",
    ],
)
