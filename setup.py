from pathlib import Path
from setuptools import setup, find_packages


def readme():
    return Path("README.md").read_text(encoding="utf-8")

setup(
    name="autofe-vsu-project",
    version="0.0.3",
    author="Daria Palchikova",
    author_email="daria.palchikova@gmail.com",
    license="MIT",
    description="Automated Feature Engineering framework",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/deola-q/AutoFE",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="feature engineering, automated feature engineering, sklearn",
    project_urls={
        "Bug Reports": "https://github.com/deola-q/AutoFE/issues",
        "Source": "https://github.com/deola-q/AutoFE",
    },
    python_requires=">=3.8",
    include_package_data=True,
    license_files=("LICENSE",),
)