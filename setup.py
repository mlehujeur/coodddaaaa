import setuptools
import os
import subprocess

class MakeTheDoc(setuptools.Command):
    description = "Generate Documentation Pages using Sphinx"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(
            ['cd docs && '
             'ln -sf ../examples/*.ipynb . &&'
             'make clean && '
             'make html '
             ''], shell=True)
        
setuptools.setup(
    name="coodddaaaa",
    version="1.3",
    author="Maximilien Lehujeur / Pierric Mora",
    author_email="maximilien.lehujeur@univ-eiffel.fr",
    description="Coda stretching",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Operating System :: Microsoft :: Windows",
        ],
    cmdclass={
        'doc': MakeTheDoc,
        },
    python_requires='>=3.7',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 
        'jupyter', 'notebook',
        'sphinx', 'sphinx-rtd-theme', 'myst-parser',  # to generate the sphinx doc
        ],
    scripts=[])
