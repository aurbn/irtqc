import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="irtqc",
    version="0.0.1",
    author="Anatoly Urban",
    author_email="urban@rcpcm.org",
    description="iRT peptide LCMSMS run quality control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aurbn/irtqc",
    packages=setuptools.find_packages(),
    install_requires=['pyteomics[DF]', 'lxml', 'matplotlib', 'pandas', 'tqdm', 'numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['irtqc = irtqc:main']},
)