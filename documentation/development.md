# Development

Development uses pre-commit for code linting and formatting. To setup development with pre-commit follow these steps after cloning the repository:

1. Create a virtual environment with python 3.10 or newer:

```
python3.10 -m venv venv
```

> [!NOTE]
> It may be necessary to install python 3.10 if it is not already on your system. Some system have python 3.10 as default and  'python' or 'python3' may be used instead.

2. Activate virtual environment:

```
source venv/bin/activate
```

3. Install the requiret python packages:

```
pip install -e[dev]
```

4. Install pre-commit hooks

```
pre-commit install
```

You are now ready to contribute.

5. Running CLI

The packages install a script which can be run with `CRD`. See

```
CRD --help
```

For more info on how to use the CLI.
