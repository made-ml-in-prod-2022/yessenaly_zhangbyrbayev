from setuptools import find_packages, setup


setup(
    name="ml_project_package",
    packages=find_packages(include=["ml_project_package", "ml_project_package.*"]),
    version="0.1.0",
    description="Package for train model and creating micro-service for online inference",
    author="Yessenaly Zhangbyrbayev",
    license="",
)