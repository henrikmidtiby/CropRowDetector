

.PHONY: run clean

.DEFAULT_GOAL := run

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	$(PYTHON) crop_row_detector_main.py rødsvingel/input_data/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundry True --generate_debug_images True --run_specific_tileset 16 20

run_test_all: $(VENV)/bin/activate
	$(PYTHON) crop_row_detector_main.py rødsvingel/input_data/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundry True

build_hough_cython_code:
	python setup.py build_ext --inplace

clean: 
	rm -rf $(VENV)
	rm -rf __pycache__