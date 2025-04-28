from __future__ import annotations

import os
import threading
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pybaselines import Baseline
from rasterio.enums import Resampling
from scipy.signal import find_peaks

# import hough_transform_grayscale # This is a custom implementation of the hough transform
from skimage.transform import hough_line
from tqdm.contrib.concurrent import process_map, thread_map

from crop_row_detector.orthomosaic_tiler import Tile


class CropRowDetector:
    def __init__(self):
        self.output_location: Path | None = None
        self.generate_debug_images = False
        self.tile_boundary = False
        self.threshold_level = 10
        self.expected_crop_row_distance = 20
        self.min_crop_row_angle = None
        self.max_crop_row_angle = None
        self.crop_row_angle_resolution = None
        self.run_parallel = True
        self.max_workers = os.cpu_count()
        # This class is just a crop row detector in form of a collection of functions,
        # all of the information is stored in the information class Tile.

    def ensure_parent_directory_exist(self, path: Path):
        temp_path = path.parent
        if not temp_path.exists():
            temp_path.mkdir(parents=True)
            # print(f"Created directory: {temp_path}")

    def get_debug_output_filepath(self, output_path, tile_number):
        return self.output_location.joinpath(f"debug_images/{tile_number}/{output_path}")

    def write_debug_image_to_file(self, output_path, img, tile_number):
        path = self.get_debug_output_filepath(output_path, tile_number)
        self.ensure_parent_directory_exist(path)
        cv2.imwrite(path, img)

    def write_debug_plot_to_file(self, output_path, tile_number):
        path = self.get_debug_output_filepath(output_path, tile_number)
        self.ensure_parent_directory_exist(path)
        plt.savefig(path, dpi=300)

    def apply_top_hat(self, h):
        filterSize = (1, int(self.expected_crop_row_distance))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        h = cv2.morphologyEx(h, cv2.MORPH_TOPHAT, kernel)
        return h

    def blur_image(self, h):
        # Blur image using a 5 x 1 average filter
        kernel = np.ones((5, 1), np.float32) / 5
        h = cv2.filter2D(h, -1, kernel)
        return h

    def compass_degree_angle_to_hough_rad(self, degree_angle):
        """
        Convert compass angles (0 north 90 east 180 south and 270 west) to radians in hough space.
        Since the hough angle is measured in image coordinates (origin in top left corner and y going down)
        and the angle is to the normal of the line measured on the unit circle, correcting for this
        ends opp canceling each other as we can always add 180 degrees since it is a line.
        """
        rad_angle = np.deg2rad(degree_angle)
        return rad_angle

    def apply_hough_lines(self, bw_image, debug_tile_number=None):
        number_of_angles = int(self.crop_row_angle_resolution * (self.max_crop_row_angle - self.min_crop_row_angle))
        min_rad_angle = self.compass_degree_angle_to_hough_rad(self.min_crop_row_angle)
        max_rad_angle = self.compass_degree_angle_to_hough_rad(self.max_crop_row_angle)
        tested_angles = np.linspace(min_rad_angle, max_rad_angle, number_of_angles)
        hspace, theta, distances = hough_line(bw_image, theta=tested_angles)
        hspace = hspace.astype(np.float32)
        hspace = self.normalize_array(hspace)
        hspace_blurred = self.blur_image(hspace)
        hspace_blurred = self.normalize_array(hspace_blurred)
        hspace_top_hat = self.apply_top_hat(hspace_blurred)
        hspace_top_hat = self.normalize_array(hspace_top_hat)
        if self.generate_debug_images and debug_tile_number is not None:
            self.write_debug_image_to_file("33_hough_image.png", 255 * hspace, debug_tile_number)
            self.write_debug_image_to_file("34_hough_image_blurred.png", 255 * hspace_blurred, debug_tile_number)
            self.write_debug_image_to_file("35_hough_image_tophat.png", 255 * hspace_top_hat, debug_tile_number)
        return hspace_top_hat, theta, distances

    def normalize_array(self, arr):
        _max = cv2.minMaxLoc(arr)[1]
        if _max > 0:
            arr = arr / _max
        else:
            # This is implemented to stop the padding tiles from being 0
            # and therefore throwing an error when using np.log, as log(0) is undefined.
            arr = arr + 10e-10
        return arr

    def determine_dominant_direction(self, hspace, theta, debug_tile_number=None):
        baseline_fitter = Baseline(theta * 180 / np.pi, check_finite=False)
        # There are 4 different ways to determine the dominant row, as seen below.
        direction_response = np.sum(np.square(hspace), axis=0)
        log_direc = np.log(direction_response)
        direc_baseline = direction_response - baseline_fitter.mor(direction_response, half_window=30)[0]
        log_direc_baseline = log_direc - baseline_fitter.mor(log_direc, half_window=30)[0]
        direction_with_most_energy_idx = np.argmax(direc_baseline)
        direction = theta[direction_with_most_energy_idx]
        if self.generate_debug_images and debug_tile_number is not None:
            self.plot_direction_energies(
                theta, log_direc, direc_baseline, log_direc_baseline, direction_response, debug_tile_number
            )
        return direction, direction_with_most_energy_idx

    def plot_direction_energies(
        self, theta, log_direc, direc_baseline, log_direc_baseline, direction_response, tile_number
    ):
        plt.figure(figsize=(16, 9))
        self.plot_direction_response_and_maximum(theta, log_direc, "blue", "log of direction response")
        self.plot_direction_response_and_maximum(theta, direc_baseline, "green", "direction response - baseline")
        self.plot_direction_response_and_maximum(
            theta, log_direc_baseline, "orange", "log of direction response - baseline"
        )
        self.plot_direction_response_and_maximum(theta, direction_response, "red", "direction response")
        plt.legend()
        self.write_debug_plot_to_file("36_direction_energies.png", tile_number)
        plt.close()

    def plot_direction_response_and_maximum(self, theta, direction_response, color, label):
        plt.plot(theta * 180 / np.pi, direction_response, color=color, label=label)
        plt.axvline(x=theta[np.argmax(direction_response)] * 180 / np.pi, color=color, linestyle="dashed")

    def determine_offsets_of_crop_rows(self, hspace, direction_idx, debug_tile_number=None):
        signal = hspace[:, direction_idx]
        peaks, _ = find_peaks(signal, distance=self.expected_crop_row_distance / 2, prominence=0.01)
        if self.generate_debug_images and debug_tile_number is not None:
            self.plot_row_offset(signal, debug_tile_number)
            self.plot_row_offset_with_peaks(signal, peaks, debug_tile_number)
        return peaks

    def plot_row_offset(self, signal, tile_number):
        plt.figure(figsize=(16, 9))
        plt.plot(signal, color="blue")
        self.write_debug_plot_to_file("38_row_offsets.png", tile_number)
        plt.close()

    def plot_row_offset_with_peaks(self, signal, peaks, tile_number):
        plt.figure(figsize=(16, 9))
        plt.plot(signal)
        plt.plot(peaks, signal[peaks], "x")
        plt.plot(np.zeros_like(signal), "--", color="gray")
        self.write_debug_plot_to_file("39_row_offsets_with_detected_peaks.png", tile_number)
        plt.close()

    def determine_line_ends_of_crop_rows(self, distances, peaks, direction, image_shape):
        vegetation_lines = []
        prev_peak_dist = 0
        for peak_idx in peaks:
            dist = distances[peak_idx]
            vegetation_lines.extend(
                self.fill_in_gaps_in_detected_crop_rows(dist, prev_peak_dist, direction, image_shape)
            )
            line_ends = self.get_line_ends_within_image(dist, direction, image_shape)
            prev_peak_dist = dist
            vegetation_lines.append(line_ends)
        valid_vegetation_lines = []
        for line_ends in vegetation_lines:
            if len(line_ends) == 2:
                valid_vegetation_lines.append(line_ends)
        return valid_vegetation_lines

    def draw_detected_crop_rows_on_image(self, vegetation_lines, plot_image, bw_image, debug_tile_number=None):
        for line_ends in vegetation_lines:
            plot_image = self.draw_crop_row(plot_image, line_ends)
        if self.generate_debug_images and debug_tile_number is not None:
            inverse_bw_image = 255 - bw_image
            for line_ends in vegetation_lines:
                inverse_bw_image = self.draw_crop_row(inverse_bw_image, line_ends)
            self.write_debug_image_to_file("40_detected_crop_rows.png", plot_image, debug_tile_number)
            self.write_debug_image_to_file(
                "45_detected_crop_rows_on_segmented_image.png", inverse_bw_image, debug_tile_number
            )
        return plot_image

    def draw_crop_row(self, image, line_ends):
        image = np.astype(image, np.uint8)  # without this opencv gives errors when trying to draw.
        cv2.line(image, (line_ends[0][0], line_ends[0][1]), (line_ends[1][0], line_ends[1][1]), (0, 0, 255), 1)
        return image

    def add_boundary_and_number_to_tile(self, image, boundary, tile_number):
        image = np.astype(image, np.uint8)  # without this opencv gives errors when trying to draw.
        c1, c2, r1, r2 = boundary
        cv2.line(image, (c1, r1), (c2 - 1, r1), (0, 0, 255), 1)
        cv2.line(image, (c1, r2 - 1), (c2 - 1, r2 - 1), (0, 0, 255), 1)
        cv2.line(image, (c1, r1), (c1, r2 - 1), (0, 0, 255), 1)
        cv2.line(image, (c2 - 1, r1), (c2 - 1, r2 - 1), (0, 0, 255), 1)
        cv2.putText(
            image,
            str(tile_number),
            (c1 + 10, r1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    def fill_in_gaps_in_detected_crop_rows(self, dist, prev_peak_dist, angle, image_shape):
        vegetation_lines = []
        if prev_peak_dist != 0:
            while dist - prev_peak_dist > 2 * self.expected_crop_row_distance:
                prev_peak_dist += self.expected_crop_row_distance
                line_ends = self.get_line_ends_within_image(prev_peak_dist, angle, image_shape)
                vegetation_lines.append(line_ends)
        return vegetation_lines

    def get_line_ends_within_image(self, dist, angle, image_shape):
        x_val_range = np.array((0, image_shape[1]))
        y_val_range = np.array((0, image_shape[0]))
        # x * cos(t) + y * sin(t) = r
        line_ends = []
        if np.sin(angle) == 0:
            y0, y1 = y_val_range
        else:
            y0, y1 = (dist - x_val_range * np.cos(angle)) / np.sin(angle)
        if np.cos(angle) == 0:
            x0, x1 = x_val_range
        else:
            x0, x1 = (dist - y_val_range * np.sin(angle)) / np.cos(angle)
        if int(y0) >= 0 and int(y0) <= image_shape[0]:
            line_ends.append([0, int(y0)])
        if int(x0) >= 0 and int(x0) <= image_shape[1]:
            line_ends.append([int(x0), 0])
        if int(x1) >= 0 and int(x1) <= image_shape[1]:
            line_ends.append([int(x1), image_shape[0]])
        if int(y1) >= 0 and int(y1) <= image_shape[0]:
            line_ends.append([image_shape[1], int(y1)])
        try:
            if line_ends[0][0] > line_ends[1][0]:
                line_ends = [line_ends[1], line_ends[0]]
        except IndexError:
            pass
        return line_ends

    def measure_vegetation_coverage_in_crop_row(self, tile, segmented_image, plot_image, vegetation_lines, direction):
        # 1. Blur image with a uniform kernel
        # Approx distance between crop rows is 16 pixels.
        # I would prefer to have a kernel size that is not divisible by two.
        vegetation_map = cv2.blur(segmented_image.astype(np.uint8), (10, 10))
        missing_vegetation_df = self.find_vegetation_in_crop_row(tile, vegetation_map, vegetation_lines, direction)
        if self.generate_debug_images and tile.tile_number is not None:
            filename = self.get_debug_output_filepath("68_vegetation_samples.csv", tile.tile_number)
            missing_vegetation_df.to_csv(filename, index=False)
            self.write_debug_image_to_file("60_vegetation_map.png", vegetation_map, tile.tile_number)
            self.write_debug_image_to_file("67_missing_plants_in_crop_line.png", plot_image, tile.tile_number)
        return missing_vegetation_df

    def find_vegetation_in_crop_row(self, tile, vegetation_map, vegetation_lines, direction):
        missing_vegetation_df_list = []
        for row_number, crop_row in enumerate(vegetation_lines):
            x_sample_coords, y_sample_coords = self.calculate_x_and_y_sample_cords_along_crop_row(crop_row, direction)
            vegetation_samples = cv2.remap(
                vegetation_map, x_sample_coords.astype(np.float32), y_sample_coords.astype(np.float32), cv2.INTER_LINEAR
            )
            missing_veg_df = pd.DataFrame(
                {
                    "tile": tile.tile_number,
                    "row": row_number,
                    "x": tile.ulc_global[0] + tile.resolution[0] * x_sample_coords,
                    "y": tile.ulc_global[1] - tile.resolution[1] * y_sample_coords,
                    "vegetation": vegetation_samples.transpose()[0],
                }
            )
            missing_vegetation_df_list.append(missing_veg_df)
        if missing_vegetation_df_list:
            missing_vegetation_df = pd.concat(missing_vegetation_df_list)
        else:
            missing_vegetation_df = pd.DataFrame({"tile": [], "row": [], "x": [], "y": [], "vegetation": []})
        return missing_vegetation_df

    def plot_points_without_vegetation_on_crop_row(self, tile, image, missing_vegetation_df):
        image = np.astype(image, np.uint8)  # without this opencv gives errors when trying to draw.
        threshold_vegetation = 60
        missing_plants = missing_vegetation_df[missing_vegetation_df["vegetation"] < threshold_vegetation]
        for _, location in missing_plants.iterrows():
            cv2.circle(
                image,
                (
                    int((location["x"] - tile.ulc_global[0]) / tile.resolution[0]),
                    int((tile.ulc_global[1] - location["y"]) / tile.resolution[1]),
                ),
                2,
                (255, 255, 0),
                -1,
            )
        return image

    def create_data_structure_containing_crop_row(
        self, tile, vegetation_map, counter, x_sample_coords, y_sample_coords
    ):
        vegetation_samples = cv2.remap(
            vegetation_map, x_sample_coords.astype(np.float32), y_sample_coords.astype(np.float32), cv2.INTER_LINEAR
        )
        crop_df = pd.DataFrame(
            {
                "tile": tile.tile_number,
                "row": counter,
                "x": tile.ulc_global[0] + tile.resolution[0] * x_sample_coords,
                "y": tile.ulc_global[1] - tile.resolution[1] * y_sample_coords,
                "vegetation": vegetation_samples.transpose()[0],
            }
        )
        return crop_df

    def calculate_x_and_y_sample_cords_along_crop_row(self, crop_row, direction):
        # Determine sample locations along the crop row
        start_point = (crop_row[0][0], crop_row[0][1])
        end_point = (crop_row[1][0], crop_row[1][1])
        distance = np.linalg.norm(np.asarray(start_point) - np.asarray(end_point))
        distance_between_samples = 1
        n_samples = np.ceil(0.0001 + distance / distance_between_samples)
        assert n_samples > 0, "n_samples is less than 0"
        x_close_to_end = start_point[0] + distance * np.sin(direction)
        y_close_to_end = start_point[1] + distance * np.cos(direction) * (-1)
        # In some cases the given angle points directly away from the end point, instead of
        # point towards the end point from the starting point. In that case, reverse the direction.
        if np.abs(x_close_to_end - end_point[0]) + np.abs(y_close_to_end - end_point[1]) > 5:
            direction = direction + np.pi
        x_sample_coords = start_point[0] + range(0, int(n_samples)) * np.sin(direction) * (1)
        y_sample_coords = start_point[1] + range(0, int(n_samples)) * np.cos(direction) * (-1)
        return x_sample_coords, y_sample_coords

    def convert_segmented_image_to_bw(self, image):
        bw_image = np.where(image < self.threshold_level, 255, 0)
        return bw_image

    def prepare_csv_files(self):
        # if csv files already exists give error
        if os.path.isfile(self.output_location.joinpath("row_information.csv")):
            raise FileExistsError("row_information.csv exists. Choose another output location or remove the file.")
        else:
            df = pd.DataFrame(
                [], columns=["tile", "x_position", "y_position", "angle", "row", "x_start", "y_start", "x_end", "y_end"]
            )
            df.to_csv(self.output_location.joinpath("row_information.csv"), index=False)
        if os.path.isfile(self.output_location.joinpath("row_information_global.csv")):
            raise FileExistsError(
                "row_information_global.csv exists. Choose another output location or remove the file."
            )
        else:
            df = pd.DataFrame(
                [],
                columns=[
                    "tile",
                    "x_position",
                    "y_position",
                    "angle",
                    "row",
                    "x_start",
                    "y_start",
                    "x_end",
                    "y_end",
                    "x_mid",
                    "y_mid",
                ],
            )
            df.to_csv(self.output_location.joinpath("row_information_global.csv"), index=False)
        if os.path.isfile(self.output_location.joinpath("points_in_rows.csv")):
            raise FileExistsError("points_in_rows.csv exists. Choose another output location or remove the file.")
        else:
            df = pd.DataFrame([], columns=["tile", "row", "x", "y", "vegetation"])
            df.to_csv(self.output_location.joinpath("points_in_rows.csv"), index=False)

    def detect_crop_rows_on_tiles_with_threads(self, segmented_ortho_tiler, plot_ortho_tiler, save_tiles=False):
        segmented_tiles = segmented_ortho_tiler.tiles
        plot_tiles = plot_ortho_tiler.tiles
        self.prepare_csv_files()
        if self.max_workers is None:
            self.max_workers = 1
        read_segmented_lock = threading.Lock()
        read_plot_lock = threading.Lock()
        write_lock = threading.Lock()
        row_info_lock = threading.Lock()
        row_info_global_lock = threading.Lock()
        row_vegetation_lock = threading.Lock()
        process_lock = threading.Lock()
        output_filename = self.output_location.joinpath("orthomosaic.tiff")
        with (
            rasterio.open(plot_ortho_tiler.orthomosaic) as plot_src,
            rasterio.open(segmented_ortho_tiler.orthomosaic) as segmented_src,
        ):
            profile = plot_src.profile
            overview_factors = plot_src.overviews(plot_src.indexes[0])
            with rasterio.open(output_filename, "w", **profile) as dst:

                def process(segmented_tile: Tile, plot_tile: Tile) -> None:
                    with read_segmented_lock:
                        segmented_img = segmented_src.read(window=segmented_tile.window_with_overlap)
                    with read_plot_lock:
                        plot_img = plot_src.read(window=plot_tile.window_with_overlap)
                        if plot_img.shape[0] > 3:
                            mask = None
                        else:
                            mask_temp = plot_src.read_masks(window=plot_tile.window_with_overlap)
                            mask = mask_temp[0]
                            for band in range(mask_temp.shape[0]):
                                mask = mask & mask_temp[band]
                    with process_lock:
                        output_img, direction, vegetation_lines, vegetation_df = self.detect_crop_rows(
                            segmented_img, segmented_tile, plot_img, plot_tile
                        )
                    with row_info_lock:
                        self.append_to_csv_of_row_information(plot_tile, direction, vegetation_lines)
                    with row_info_global_lock:
                        self.append_to_csv_of_row_information_global(plot_tile, direction, vegetation_lines)
                    with row_vegetation_lock:
                        self.append_to_csv_vegetation_row(vegetation_df)
                    if save_tiles:
                        plot_tile.save_tile(output_img, mask, self.output_location.joinpath("tiles"))
                    output = plot_tile.get_window_pixels(output_img)
                    if mask is not None:
                        mask = plot_tile.get_window_pixels(np.expand_dims(mask, 0)).squeeze()
                    with write_lock:
                        dst.write(output, window=plot_tile.window)
                        if mask is not None:
                            dst.write_mask(mask, window=plot_tile.window)

                thread_map(process, segmented_tiles, plot_tiles, max_workers=self.max_workers)

        with rasterio.open(output_filename, "r+") as dst:
            dst.build_overviews(overview_factors, Resampling.average)

    def detect_crop_rows_on_tiles_with_process_pools(self, segmented_ortho_tiler, plot_ortho_tiler, save_tiles=False):
        segmented_tiles = segmented_ortho_tiler.tiles
        plot_tiles = plot_ortho_tiler.tiles
        results = process_map(
            partial(self.detect_crop_rows_as_process, save_tiles=save_tiles),
            segmented_tiles,
            plot_tiles,
            chunksize=1,
            max_workers=self.max_workers,
        )
        new_plot_tiles = [tile for tile, _, _, _ in results]
        directions = [direction for _, direction, _, _ in results]
        vegetation_lines_list = [veg for _, _, veg, _ in results]
        vegetation_df = pd.concat([veg_df for _, _, _, veg_df in results])

        output_filename = self.output_location.joinpath("orthomosaic.tiff")
        with (
            rasterio.open(plot_ortho_tiler.orthomosaic) as src,
        ):
            profile = src.profile
            overview_factors = src.overviews(src.indexes[0])
        with rasterio.open(output_filename, "w", **profile) as dst:
            for tile in new_plot_tiles:
                dst.write(tile.output, window=tile.window)
                if tile.output.shape[0] <= 3:
                    dst.write_mask(tile.mask, window=tile.window)
        with rasterio.open(output_filename, "r+") as dst:
            dst.build_overviews(overview_factors, Resampling.average)
        self.create_csv_of_row_information(plot_tiles, directions, vegetation_lines_list)
        self.create_csv_of_row_information_global(plot_tiles, directions, vegetation_lines_list)
        self.vegetation_row_to_csv(vegetation_df)

    def detect_crop_rows_as_process(self, segmented_tile: Tile, plot_tile: Tile, save_tiles=False):
        segmented_image, _ = segmented_tile.read_tile()
        plot_image, plot_mask = plot_tile.read_tile()
        mask = plot_mask[0]
        for band in range(plot_mask.shape[0]):
            mask = mask & plot_mask[band]
        output_img, direction, vegetation_lines, vegetation_df = self.detect_crop_rows(
            segmented_image, segmented_tile, plot_image, plot_tile
        )
        if save_tiles:
            plot_tile.save_tile(output_img, mask, self.output_location.joinpath("tiles"))
        output = plot_tile.get_window_pixels(output_img)
        mask = plot_tile.get_window_pixels(np.expand_dims(mask, 0)).squeeze()
        plot_tile.output = output
        plot_tile.mask = mask
        return plot_tile, direction, vegetation_lines, vegetation_df

    def detect_crop_rows(self, segmented_image, segmented_tile, plot_image_original, plot_tile):
        assert segmented_image.shape[0] == 1, "The segmented image has more then one color channel."
        assert plot_tile.ulc == segmented_tile.ulc, "The two tiles are not the same location."
        segmented_image = np.squeeze(segmented_image)
        plot_image = np.moveaxis(plot_image_original[:3, :, :], 0, -1)  # TODO better fix for multispectral
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
        bw_tile = self.convert_segmented_image_to_bw(segmented_image)
        hspace, theta, distances = self.apply_hough_lines(bw_tile, debug_tile_number=segmented_tile.tile_number)
        direction, direction_idx = self.determine_dominant_direction(
            hspace, theta, debug_tile_number=segmented_tile.tile_number
        )
        peaks = self.determine_offsets_of_crop_rows(hspace, direction_idx, debug_tile_number=segmented_tile.tile_number)
        vegetation_lines = self.determine_line_ends_of_crop_rows(distances, peaks, direction, segmented_image.shape)
        plot_image = self.draw_detected_crop_rows_on_image(
            vegetation_lines, plot_image, bw_tile, debug_tile_number=segmented_tile.tile_number
        )
        vegetation_df = self.measure_vegetation_coverage_in_crop_row(
            segmented_tile, segmented_image, plot_image, vegetation_lines, direction
        )
        plot_image = self.plot_points_without_vegetation_on_crop_row(segmented_tile, plot_image, vegetation_df)
        if self.tile_boundary:
            boundary = plot_tile.get_window_pixels_boundary()
            plot_image = self.add_boundary_and_number_to_tile(
                plot_image, boundary, tile_number=segmented_tile.tile_number
            )

        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
        plot_image_original[:3, :, :] = np.moveaxis(plot_image, -1, 0)
        return plot_image_original, direction, vegetation_lines, vegetation_df

    def append_to_csv_of_row_information(self, tile: Tile, direction, vegetation_lines):
        row_information = []
        if direction < 0:
            direction = np.pi + direction
        for row_number, row in enumerate(vegetation_lines):
            row_information.append(
                [
                    tile.tile_number,
                    tile.tile_position[0],
                    tile.tile_position[1],
                    direction,
                    row_number,
                    row[0][0],
                    row[0][1],
                    row[1][0],
                    row[1][1],
                ]
            )
        row_information_df = pd.DataFrame(
            row_information,
            columns=["tile", "x_position", "y_position", "angle", "row", "x_start", "y_start", "x_end", "y_end"],
        )
        row_information_df.to_csv(
            self.output_location.joinpath("row_information.csv"), mode="a", header=False, index=False
        )

    def create_csv_of_row_information(self, tiles, directions, vegetation_lines_list):
        row_information = []
        for tile, direction, vegetation_lines in zip(tiles, directions, vegetation_lines_list, strict=False):
            if direction < 0:
                direction = np.pi + direction
            for row_number, row in enumerate(vegetation_lines):
                row_information.append(
                    [
                        tile.tile_number,
                        tile.tile_position[0],
                        tile.tile_position[1],
                        direction,
                        row_number,
                        row[0][0],
                        row[0][1],
                        row[1][0],
                        row[1][1],
                    ]
                )
        row_information_df = pd.DataFrame(
            row_information,
            columns=["tile", "x_position", "y_position", "angle", "row", "x_start", "y_start", "x_end", "y_end"],
        )
        row_information_df.to_csv(self.output_location.joinpath("row_information.csv"))

    def append_to_csv_of_row_information_global(self, tile: Tile, direction, vegetation_lines):
        row_information = []
        if direction < 0:
            direction = np.pi + direction
        for row_number, row in enumerate(vegetation_lines):
            row_information.append(
                [
                    tile.tile_number,
                    tile.tile_position[0],
                    tile.tile_position[1],
                    direction,
                    row_number,
                    tile.ulc_global[0] + tile.resolution[0] * row[0][0],
                    tile.ulc_global[1] - tile.resolution[1] * row[0][1],
                    tile.ulc_global[0] + tile.resolution[0] * row[1][0],
                    tile.ulc_global[1] - tile.resolution[1] * row[1][1],
                    (2 * tile.ulc_global[0] + tile.resolution[0] * (row[0][0] + row[1][0])) / 2,
                    (2 * tile.ulc_global[1] - tile.resolution[1] * (row[0][1] + row[1][1])) / 2,
                ]
            )
        row_information_df = pd.DataFrame(
            row_information,
            columns=[
                "tile",
                "x_position",
                "y_position",
                "angle",
                "row",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "x_mid",
                "y_mid",
            ],
        )
        row_information_df.to_csv(
            self.output_location.joinpath("row_information_global.csv"), mode="a", header=False, index=False
        )

    def create_csv_of_row_information_global(self, tiles, directions, vegetation_lines_list):
        row_information = []
        for tile, direction, vegetation_lines in zip(tiles, directions, vegetation_lines_list, strict=False):
            if direction < 0:
                direction = np.pi + direction
            for row_number, row in enumerate(vegetation_lines):
                row_information.append(
                    [
                        tile.tile_number,
                        tile.tile_position[0],
                        tile.tile_position[1],
                        direction,
                        row_number,
                        tile.ulc_global[0] + tile.resolution[0] * row[0][0],
                        tile.ulc_global[1] - tile.resolution[1] * row[0][1],
                        tile.ulc_global[0] + tile.resolution[0] * row[1][0],
                        tile.ulc_global[1] - tile.resolution[1] * row[1][1],
                        (2 * tile.ulc_global[0] + tile.resolution[0] * (row[0][0] + row[1][0])) / 2,
                        (2 * tile.ulc_global[1] - tile.resolution[1] * (row[0][1] + row[1][1])) / 2,
                    ]
                )
        DF_row_information = pd.DataFrame(
            row_information,
            columns=[
                "tile",
                "x_position",
                "y_position",
                "angle",
                "row",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "x_mid",
                "y_mid",
            ],
        )
        DF_row_information.to_csv(self.output_location.joinpath("row_information_global.csv"))

    def append_to_csv_vegetation_row(self, vegetation_df):
        csv_path = self.output_location.joinpath("points_in_rows.csv")
        vegetation_df.to_csv(csv_path, mode="a", header=False, index=False)

    def vegetation_row_to_csv(self, vegetation_df):
        csv_path = self.output_location.joinpath("points_in_rows.csv")
        vegetation_df.to_csv(csv_path, index=False)

    def save_statistics(self, segmented_orthomosaic, orthomosaic, tile_size, number_of_tiles):
        statistics_path = self.output_location.joinpath("statistics")
        self.ensure_parent_directory_exist(statistics_path.joinpath("output_file.txt"))
        print(f'Writing statistics to the folder "{ statistics_path }"')
        with open(statistics_path + "/output_file.txt", "w") as f:
            f.write("Input parameters:\n")
            f.write(f" - Segmented Orthomosaic: {segmented_orthomosaic}\n")
            f.write(f" - Orthomosaic: {orthomosaic}\n")
            f.write(f" - Tile sizes: {tile_size}\n")
            f.write(f" - Output tile location: {self.tile_location}\n")
            f.write(f" - Generated debug images: {self.generate_debug_images}\n")
            f.write(f" - Tile boundary: {self.tile_boundary}\n")
            f.write(f" - Expected crop row distance: {self.expected_crop_row_distance}\n")
            f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")
            f.write("\n\nOutput from run\n")
            f.write(f" - Number of tiles: {number_of_tiles}\n")
