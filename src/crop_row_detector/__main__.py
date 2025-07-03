from __future__ import annotations

import argparse
import os
import pathlib
from copy import deepcopy
from typing import Any

from crop_row_detector import CropRowDetector, OrthomosaicTiles


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Crop Row Detector", description="Detect crop rows in segmented image", epilog="test"
    )
    parser.add_argument("segmented_orthomosaic", help="Path to the segmented_orthomosaic that you want to process.")
    parser.add_argument(
        "--orthomosaic",
        metavar="FILENAME",
        help="Path to the orthomosaic that you want to plot on. if not set, the segmented_orthomosaic will be used.",
    )
    parser.add_argument(
        "--tile_size",
        default=500,
        nargs="+",
        type=int,
        help="The height and width of tiles that are analyzed. Default is 500.",
    )
    parser.add_argument(
        "--tile_overlap",
        default=0,
        type=float,
        help="Percentage overlap between tiles in tile size. Added as padding making the actual tile size larger.",
    )
    parser.add_argument(
        "--output_location",
        default="output/mahal",
        metavar="FILENAME",
        type=pathlib.Path,
        help="The location in which to save the mahalanobis tiles.",
    )
    parser.add_argument(
        "--save_tiles",
        action="store_true",
        help="If set tiles are saved at output_location/tiles. Useful for debugging or parameter tweaking. Default no tiles are saved.",
    )
    parser.add_argument(
        "--generate_debug_images",
        action="store_true",
        help="If set debug images will be generated. default is no debug images is generated.",
    )
    parser.add_argument(
        "--save_statistics",
        action="store_true",
        help="If set statistics are saved at output_location.",
    )
    parser.add_argument(
        "--tile_boundary",
        action="store_true",
        help="if set will plot a boundary on each tile and the tile number on the tile. Default is no boundary and tile number.",
    )
    parser.add_argument(
        "--run_specific_tile",
        nargs="+",
        type=int,
        metavar="TILE_ID",
        help="If set, only run the specific tile numbers. (--run_specific_tile 16 65) will run tile 16 and 65.",
    )
    parser.add_argument(
        "--run_specific_tileset",
        nargs="+",
        type=int,
        metavar="FROM_TILE_ID TO_TILE_ID",
        help="takes two inputs like (--from_specific_tileset 16 65). This will run every tile from 16 to 65.",
    )
    parser.add_argument(
        "--expected_crop_row_distance",
        default=20,
        type=int,
        metavar="DISTANCE",
        help="The expected distance between crop rows in pixels, default is 20.",
    )
    parser.add_argument(
        "--min_angle",
        default=0,
        type=float,
        metavar="ANGLE",
        help="The minimum angle in which the crop rows is expected. Value between 0 and 180. (In compas angles, i.e. 0 north, 90 east, 180 south and 270 west). Default is 0.",
    )
    parser.add_argument(
        "--max_angle",
        default=180,
        type=float,
        metavar="ANGLE",
        help="The maximum angle in which the crop rows is expected. Value between 0 and 180. (In compas angles, i.e. 0 north, 90 east, 180 south and 270 west). Default is 180.",
    )
    parser.add_argument(
        "--angle_resolution",
        default=8,
        type=int,
        metavar="BINS",
        help="How many bins each degree is divided into. Default is 8.",
    )
    parser.add_argument(
        "--max_workers",
        default=os.cpu_count(),
        type=int,
        help="Set the maximum number of workers. Default to number of cpus.",
    )
    parser.add_argument(
        "--use_process_pools",
        action="store_true",
        help="Use process pools instead of threads. This may come at an extra cost of memory but will be faster.",
    )
    return parser


def _parse_args(args: Any = None) -> Any:
    parser = _get_parser()
    return parser.parse_args(args)


def _create_output_location(output_directory: pathlib.Path) -> None:
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)


def init_tile_separator(args):
    if isinstance(args.tile_size, int):
        tile_size = args.tile_size
    elif len(args.tile_size) == 1:
        tile_size = args.tile_size[0]
    elif len(args.tile_size) == 2:
        tile_size = tuple(args.tile_size)
    else:
        raise Exception("Tiles size must be 1 or 2 integers.")
    # Initialize the tile separator
    segmented_tiler = OrthomosaicTiles(
        orthomosaic=args.segmented_orthomosaic,
        tile_size=tile_size,
        overlap=args.tile_overlap,
        run_specific_tile=args.run_specific_tile,
        run_specific_tileset=args.run_specific_tileset,
    )
    segmented_tiler.divide_orthomosaic_into_tiles()
    if args.orthomosaic is None:
        plot_tiler = deepcopy(segmented_tiler)
    else:
        plot_tiler = OrthomosaicTiles(
            orthomosaic=args.orthomosaic,
            tile_size=tile_size,
            overlap=args.tile_overlap,
            run_specific_tile=args.run_specific_tile,
            run_specific_tileset=args.run_specific_tileset,
        )
        plot_tiler.divide_orthomosaic_into_tiles()
    return segmented_tiler, plot_tiler, tile_size


def run_crop_row_detector(segmented_tiler, plot_tiler, tile_size, args):
    # Initialize the crop row detector
    crd = CropRowDetector()
    crd.output_location = args.output_location
    crd.generate_debug_images = args.generate_debug_images
    crd.tile_boundary = args.tile_boundary
    crd.expected_crop_row_distance = args.expected_crop_row_distance
    crd.min_crop_row_angle = args.min_angle
    crd.max_crop_row_angle = args.max_angle
    crd.crop_row_angle_resolution = args.angle_resolution
    crd.threshold_level = 12
    crd.max_workers = args.max_workers
    if args.use_process_pools:
        crd.detect_crop_rows_on_tiles_with_process_pools(segmented_tiler, plot_tiler, save_tiles=args.save_tiles)
    else:
        crd.detect_crop_rows_on_tiles_with_threads(segmented_tiler, plot_tiler, save_tiles=args.save_tiles)
    if args.save_statistics:
        crd.save_statistics(args.segmented_orthomosaic, args.orthomosaic, tile_size, len(plot_tiler.tiles))


def _main():
    args = _parse_args()
    _create_output_location(args.output_location)
    segmented_tiler, plot_tiler, tile_size = init_tile_separator(args)
    run_crop_row_detector(segmented_tiler, plot_tiler, tile_size, args)


if __name__ == "__main__":
    _main()

# python3 crop_row_detector_main.py rødsvingel/input_data/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundary True --generate_debug_images True --run_specific_tile 16
# gdal_merge.py -o rødsvingel/rødsvingel_crd.tif -a_nodata 255 rødsvingel/tiles_crd/mahal*.tiff
