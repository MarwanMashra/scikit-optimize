try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.5 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the
# main skopt __init__ can detect if it is being loaded by the setup
# routine
builtins.__SKOPT2_SETUP__ = True

import skopt2

VERSION = skopt2.__version__

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]


setup(
    name="scikit-optimize2",
    version=VERSION,
    description="Sequential model-based optimization toolbox.",
    long_description=open("README.rst").read(),
    url="https://github.com/MarwanMashra/scikit-optimize",
    license="BSD 3-clause",
    author="The scikit-optimize contributors",
    classifiers=CLASSIFIERS,
    packages=[
        "skopt2",
        "skopt2.learning",
        "skopt2.optimizer",
        "skopt2.space",
        "skopt2.learning.gaussian_process",
        "skopt2.sampler",
    ],
    install_requires=[
        "joblib>=0.11",
        "pyaml>=16.9",
        "numpy>=1.13.3",
        "scipy>=0.19.1",
        "scikit-learn>=0.20.0",
    ],
    extras_require={"plots": ["matplotlib>=2.0.0"]},
)
