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

### Visualize raw binary file
Activate your environment and type `viewephys`, you can then load a neuropixel binary file using the file menu.

### Load in a numpy array or slice
```python
# if running ipython, you may have to use the `%gui qt` magic command
import numpy as np
from viewephys.gui import viewephys
nc, ns, fs = (384, 50000, 30000)  # this mimics one second of neuropixel data
data = np.random.randn(nc, ns) / 1e6  # volts by default
ve = viewephys(data, fs=fs)
```

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
