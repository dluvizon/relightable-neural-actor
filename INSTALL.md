# Installation of Relightable Neural Actor

1. First, create a _conda_ environment:
```sh
conda env create -f environment.yml
```

You can make sure your Pytorch installation is working properly with GPU support with ():
```python
conda activate relitactor
python -c 'import torch; print(torch.cuda.is_available())'
# The output should be "True".
```

2. Install [OpenDR](https://ps.is.mpg.de/code/opendr) (snapshot provided in this repo)
```sh
pushd opendr
python setup.py install
popd
```

3. Install this project with pip
```sh
python -m pip install --editable .
```
