from setuptools import setup, find_packages

VERSION = "0.0.0.1.3.1"
DESCRIPTION = ""
LONG_DESCRIPTION = (
    "used to combine all tt predictors in one class"
)

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="tt_predictor_backend",
    version=VERSION,
    author="Yuqi Li",
    author_email="yuqil@aipaca-corp.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
