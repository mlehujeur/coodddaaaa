import setuptools


setuptools.setup(
    name="coodddaaaa",
    version="1.0",
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
    python_requires='>=3.7',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 
        'jupyter', 'notebook',
        ],
    scripts=[])
