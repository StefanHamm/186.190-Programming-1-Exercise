# 186.190-Programming-1-Exercise

# Creating conda environment
```bash
conda env create -f environment.yml
```

# Activating conda environment
```bash
conda activate tran-optim
```

# Installing perl with winget
```bash
winget install --id=StrawberryPerl.StrawberryPerl -e
```

# install latex classes
```bash
tlmgr update --self
tlmgr install standalone
```

# Building the docker image
Should build it automatically when you run the visualization function.
```bash
docker build -t tran-optim .
```