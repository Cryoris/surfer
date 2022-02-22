import inspect
import sys
import setuptools

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

if not hasattr(setuptools, "find_namespace_packages") or not inspect.ismethod(
    setuptools.find_namespace_packages
):
    print(
        "Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
        "Upgrade it to version >='40.1.0' and repeat install.".format(
            setuptools.__version__
        )
    )
    sys.exit(1)

setuptools.setup(
    name="surfer",
    version="0.1.0",
    description="Quantum Circuits: Gradients and Quantum Fisher Information",
    author="Julien Gacon",
    author_email="julien.gacon@epfl.ch",
    license="Apache-2.0",
    classifiers=(
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ),
    keywords=["qiskit", "quantum circuit gradients", "quantum fisher information"],
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.6",
    zip_safe=False,
)
