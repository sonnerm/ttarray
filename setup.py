from setuptools import setup,find_packages
with open('README.rst') as f:
    long_description=f.read()
setup(
    name="ttarray",
    packages=find_packages(),
    install_requires=["numpy>=1.19.0"],
    version="0.0.1",
    url="https://github.com/sonnerm/ttarray",
    author="Michael Sonner",
    author_email="sonnerm@gmail.com",
    description="Library for tensor train arrays with numpy-compatible api",
    license="MIT",
    long_description=long_description,
    license_files=["LICENSE"],
    platforms="any",
    # extras_require={"sparse linalg for dmrg":["scipy"],"interoperability with tenpy":["physics-tenpy"],"interoperability with quimb":["quimb"],"storing as hdf5":["hdf5"]},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning"
    ],
)
