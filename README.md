# Orthomosaic Crop Row Detector

The Orthomosaic Crop Row Detector can be used to detect crop rows in orthomosaics. It uses the original orthomosaic and a gray-scale orthomosaic of the crops locations to determine crops rows. The orthomosaic is divided into tiles and crop rows within a tile is assumed to have the same orientation.

See [Documentation](https://henrikmidtiby.github.io/CropRowDetector/index.html) for more info.

# Installation

Development uses pre-commit for code linting and formatting. To setup development with pre-commit follow these steps after cloning the repository:

1. Create a virtual environment with python 3.11 or newer:

```
python3.11 -m venv venv
```

> [!NOTE]
> It may be necessary to install python 3.11 if it is not already on your system. Some system have python 3.11 as default and  'python' or 'python3' may be used instead.

2. Activate virtual environment:

```
source venv/bin/activate
```

3. Install the required python packages:

```
pip install .
```

See the [install documentation](https://henrikmidtiby.github.io/CropRowDetector/installation.html) for more info on installation.

## Development

To get started on development see the [development documentation](https://henrikmidtiby.github.io/CropRowDetector/contributing.html).
