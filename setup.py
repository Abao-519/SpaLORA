from setuptools import setup, find_packages

setup(
    name="SpaLORA",
    description="SpaLORA: A Graph Neural Network Framework for Spatial Multi-Omics Integration Emphasizing Low-Expression Gene Signals",
    author="Changhe Li",
    author_email="lch@nefu.edu.cn",
    license="MIT",
    packages=find_packages(),
    install_requires=["requests",],
    include_package_data=True,
    zip_safe=False,
)
