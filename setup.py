from setuptools import setup
setup(
    name="justmps",
    packages=["justmps"],
    install_requires=["numpy"],
    version="0.0.1",
    # url="https://github.com/sonnerm/justmps",
    author="Michael Sonner",
    author_email="sonnerm@gmail.com",
    description="Library for matrix product states which are fully numpy compatible",
    extras_require={"sparse linalg for dmrg":["scipy"],"interop with tenpy":["physics-tenpy"],"interop with quimb":["quimb"],"storing as hdf5":["hdf5"]},
)
