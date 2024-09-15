.PHONY: run clean

.DEFAULT_GOAL := run


ifeq ($(OS), Windows_NT)
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

$(VENV)/Scripts/activate: requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

run: $(VENV)/Scripts/activate
	$(PYTHON) crop_row_detector_main.py Tests/rodsvingel/input_data/rodsvingel.tif --orthomosaic Tests/rodsvingel/input_data/2023-04-03_Rodsvingel_1._ars_Wagner_JSJ_2_ORTHO.tif --output_tile_location Tests/rodsvingel/tiles_crd --tile_size 500 --tile_boundry True --generate_debug_images True --run_parralel "" --run_specific_tileset 16 20

clean:
	if exist "./$(VENV)" rd /s /q $(VENV)
	if exist "./__pycache__" rd /s /q __pycache__

else

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

run_83_98: $(VENV)/bin/activate
	$(PYTHON) crop_row_detector_main.py rødsvingel/input_data/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundry True --generate_debug_images True --run_specific_tileset 83 98

run_6_7_20_21: $(VENV)/bin/activate
	$(PYTHON) crop_row_detector_main.py rødsvingel/input_data/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundry True --generate_debug_images True --run_specific_tile 6 7 20 21

build_hough_cython_code:
	python setup.py build_ext --inplace

clean: 
	rm -rf $(VENV)
	rm -rf __pycache__
endif