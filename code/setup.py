from setuptools import setup, find_packages

setup(
    name="audio_dm_training",
    version="0.0",
    packages=find_packages(),
    install_requires=[],
    author="Daniel Bin Schmid",
    author_email="danielbin.schmid@tum.de",
    description="",
    license="",
    keywords="",
    entry_points={
        "console_scripts": ["dm = experiments.dm:main"],  # CLI entry point
    },
)
