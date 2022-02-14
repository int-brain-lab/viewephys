# viewephys
Neuropixel raw data viewer

## Installation
`pip install viewephys`
This is compatible with the [IBL environment](https://github.com/int-brain-lab/iblenv)

Alternatively, for developers:
```
git clone https://github.com/oliche/viewephys.git
cd viewephys
pip install -e .
```

## Examples
Activate your environment and type `viewephys`, you can then load a neuropixel binary file using the file menu.

## Contribution
Fork and PR.

Pypi Release checklist:
```shell
flake8
rm -fR dist
rm -fR build
python setup.py sdist bdist_wheel
twine upload dist/*
#twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
