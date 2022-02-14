import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]

setuptools.setup(
    name="viewephys",
    version="0.0.0",
    author="Olivier Winter",
    description="Raw Neuropixel data viewer for numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oliche/viewephys",
    project_urls={
        "Bug Tracker": "https://github.com/oliche/viewephys/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=require,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': ['viewephys=viewephys.command_line:viewephys'],
    }
)
