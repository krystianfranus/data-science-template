from setuptools import find_packages, setup

with open("VERSION", "r") as f:
    version = f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().split("\n")

setup(
    name="src",
    version=version,
    author="Krystian Franus",
    author_email="krystian.franus@gmail.com",
    description="Short description",
    long_description=long_description,
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
)
