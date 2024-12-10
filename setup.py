from setuptools import setup, find_packages

setup(
    name="perf_dsa",
    version="0.0.1",
    author="Ruisheng Su",
    author_email="r.su@tue.nl",
    description="perfDSA: digital subtraction perfusion angiography using deconvolution",
    long_description=open("README.md").read(),
    url="https://github.com/RuishengSu/perfDSA",
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Modified MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)

