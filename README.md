# 186.190-Programming-1-Exercise

# Run the program
```bash
python src/construction.py [OPTIONS]
```

#### Options
| Option            | Description                      | Default           |
|-------------------|----------------------------------|-------------------|
| -t or --track     | Path to the racetrack file       | tracks/track_02.t |
| -o or --output    | Path to the output file          | routes/output.csv |
| -v or --visualize | Enable graphical visualization   | Disabled          |
 | -d or --d         | Maximum depth for graph builidng | 1                 |

# Benchmarking

Note: If having trouble running benchmark due to module not found error, add current dir to python path: `export PYTHONPATH=.`

```bash
python -m src.benchmark [--target {bfs, construction}] [--track TRACK_FILE] [--depth DEPTH] [--values VALUES] [--processes PROCESSES] [--warmups WARMUPS]
```

for simple speed up use `--fast`

example usage for track 02, construction with depth 3:

```bash
python -m src.benchmark --target construction --track track_02.t --depth 3
```

solution files are dumped to benchmark dir

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