from setuptools import setup
setup(
    name="ttarray",
    packages=["ttarray"],
    install_requires=["numpy"],
    version="0.0.1",
    # url="https://github.com/sonnerm/ttarray",
    author="Michael Sonner",
    author_email="sonnerm@gmail.com",
    description="Library for tensor train arrays which are compatible to numpy ndarray",
    extras_require={"sparse linalg for dmrg":["scipy"],"interoperability with tenpy":["physics-tenpy"],"interoperability with quimb":["quimb"],"storing as hdf5":["hdf5"]},
)
