"""
    Module to set up light house template build pipeline.
"""
# Standard modules
import os
import sys
from setuptools import setup, find_packages, Distribution


class PlatformSpecificDistribution(Distribution):
    """
    Distribution which always forces a binary package with platform name
    """

    def has_ext_modules(self):
        """Overrides super method to force platform-specific name for package

        Returns:
            bool: _description_
        """
        return True


# Setting up version for python.
if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

package_version = os.environ.get("VERSION")

if os.environ.get("BUILD_TAG") == "alpha":
    package_version = package_version + "a"
elif os.environ.get("BUILD_TAG") == "beta":
    package_version = package_version + "b"

# Packages not being installed in build
setup(
    name="databricks-ci-cd-test",
    version=package_version,
    author="Lucas Griva",
    author_email="lgriva@piconsulting.com.ar",
    description="databricks ci/cd test",
    url="https://github.com/lgriva-ext/ml_solution",
    license="Other/Proprietary 2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    # Refer to https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        "pandas==1.4.2",
        "numpy==1.21.5",
        "databricks-feature-store==0.10.0",
        "scikit-learn==1.0.2",
        "mlflow==2.1.1",
        "pyspark==3.3.2.dev0",
    ],
    dependency_links=["https://pypi.org/simple"],
    platforms=["Windows", "Linux", "MacOS"],
    test_suite="pytest",
    python_requires=">=3.8",
    distclass=PlatformSpecificDistribution,
)
