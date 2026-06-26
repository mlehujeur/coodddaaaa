import setuptools
import os
import subprocess


def load_version_number():
    version_file = os.path.join('coodddaaaa', 'version.py')
    if not os.path.isfile(version_file):
        raise IOError(version_file)

    with open(version_file, "r") as fid:
        for line in fid:
            if line.strip('\n').strip().startswith('__version__'):
                vernum = line.strip('\n').split('=')[-1].split()[0].strip().strip('"').strip("'")
                break
        else:
            raise Exception(f'could not detect __version__ affectation in {version_file}')
    return vernum


class MakeTheDoc(setuptools.Command):
    description = "Generate Documentation Pages using Sphinx"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(
            ['sphinx-build docs/ docs/_build'], shell=True)
        
__version__ = load_version_number()        
setuptools.setup(
    name="coodddaaaa",
    version=__version__,
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
        'nbsphinx',
        ],
    scripts=[])
