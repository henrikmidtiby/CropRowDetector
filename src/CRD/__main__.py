from __future__ import annotations

import argparse
import os

from CRD.crop_row_detector import crop_row_detector
from CRD.orthomosaic_tiler import OrthomosaicTiles


def parse_cmd_arguments():
    parser = argparse.ArgumentParser(description="Detect crop rows in segmented image")
    parser.add_argument("segmented_orthomosaic", help="Path to the segmented_orthomosaic that you want to process.")
    parser.add_argument(
        "--orthomosaic",
        metavar="FILENAME",
        help="Path to the orthomosaic that you want to plot on. if not set, the segmented_orthomosaic will be used.",
    )
    parser.add_argument(
        "--tile_size",
        default=3000,
        type=int,
        help="The height and width of tiles that are analyzed. Default is %(default).",
    )
    parser.add_argument(
        "--output_tile_location",
        default="output/mahal",
        metavar="FILENAME",
        help="The location in which to save the mahalanobis tiles.",
    )
    parser.add_argument(
        "--generate_debug_images",
        action="store_true",
        help="If set debug images will be generated. default is no debug images is generated.",
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
        help="The expected distance between crop rows in pixels, default is %(default).",
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
        "--run_single_thread",
        action="store_false",
        help="If set the program will run in as a single thread. Default is to run in parallel.",
    )
    parser.add_argument(
        "--max_workers",
        default=os.cpu_count(),
        type=int,
        help="Set the maximum number of workers. Default to number of cpus.",
    )
    args = parser.parse_args()
    return args


def init_tile_separator(args):
    # Initialize the tile separator
    tiler = OrthomosaicTiles(
        orthomosaic=args.segmented_orthomosaic,
        tile_size=args.tile_size,
        run_specific_tile=args.run_specific_tile,
        run_specific_tileset=args.run_specific_tileset,
    )
    segmented_tile_list = tiler.divide_orthomosaic_into_tiles()
    if args.orthomosaic is None:
        plot_tile_list = segmented_tile_list.copy()
    else:
        tiler = OrthomosaicTiles(
            orthomosaic=args.orthomosaic,
            tile_size=args.tile_size,
            run_specific_tile=args.run_specific_tile,
            run_specific_tileset=args.run_specific_tileset,
        )
        plot_tile_list = tiler.divide_orthomosaic_into_tiles()
    return segmented_tile_list, plot_tile_list


def run_crop_row_detector(segmented_tile_list, plot_tile_list, args):
    # Initialize the crop row detector
    crd = crop_row_detector()
    crd.generate_debug_images = args.generate_debug_images
    crd.tile_boundary = args.tile_boundary
    crd.expected_crop_row_distance = args.expected_crop_row_distance
    crd.min_crop_row_angle = args.min_angle
    crd.max_crop_row_angle = args.max_angle
    crd.crop_row_angle_resolution = args.angle_resolution
    crd.threshold_level = 12
    crd.run_parallel = args.run_single_thread  # true if not set e.g. run in parallel
    crd.max_workers = args.max_workers
    crd.main(segmented_tile_list, plot_tile_list, args)


def _main():
    args = parse_cmd_arguments()
    segmented_tile_list, plot_tile_list = init_tile_separator(args)
    run_crop_row_detector(segmented_tile_list, plot_tile_list, args)


if __name__ == "__main__":
    _main()

# python3 crop_row_detector_main.py rødsvingel/input_data/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundary True --generate_debug_images True --run_specific_tile 16
# gdal_merge.py -o rødsvingel/rødsvingel_crd.tif -a_nodata 255 rødsvingel/tiles_crd/mahal*.tiff
