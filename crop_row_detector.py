import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from pybaselines import Baseline
from scipy.signal import find_peaks

# import hough_transform_grayscale # This is a custom implementation of the hough transform
from skimage.transform import hough_line
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class tile_data_holder:
    def __init__(self):
        self.segmented_img = None
        self.veg_img = None
        self.veg_img_constant = None
        self.gray = None
        self.gray_inverse = None


class crop_row_detector:
    def __init__(self):
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

    def ensure_parent_directory_exist(self, path):
        temp_path = Path(path).parent
        if not temp_path.exists():
            temp_path.mkdir(parents=True)
            # print(f"Created directory: {temp_path}")

    def get_debug_output_filepath(self, output_path, tile):
        return tile.output_tile_location + "/debug_images/" + f"{tile.tile_number}" + "/" + output_path

    def write_image_to_file(self, output_path, img, tile):
        if tile.generate_debug_images:
            path = self.get_debug_output_filepath(output_path, tile)
            self.ensure_parent_directory_exist(path)
            cv2.imwrite(path, img)

    def write_plot_to_file(self, output_path, tile):
        if tile.generate_debug_images:
            path = self.get_debug_output_filepath(output_path, tile)
            self.ensure_parent_directory_exist(path)
            plt.savefig(path, dpi=300)

    def apply_top_hat(self, h, tile):
        filterSize = (1, int(tile.expected_crop_row_distance))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        h = cv2.morphologyEx(h, cv2.MORPH_TOPHAT, kernel)
        return h

    def blur_image(self, h):
        # Blur image using a 5 x 1 average filter
        kernel = np.ones((5, 1), np.float32) / 5
        h = cv2.filter2D(h, -1, kernel)
        return h

    def compas_dregree_angle_to_hough_rad(self, degree_angle):
        """
        Convert compas angles (0 north 90 east 180 south and 270 west) to radians in hough space.
        Since the hough angle is measured in image coordinates (origin in top left corner and y going down)
        and the angle is to the normal of the line measured on the unit circle, correcting for this
        ends opp canceling each other as we can always add 180 degrees since it is a line.
        """
        rad_angle = np.deg2rad(degree_angle)
        return rad_angle

    def apply_hough_lines(self, tile, tile_img_data):
        number_of_angles = int(self.crop_row_angle_resolution * (self.max_crop_row_angle - self.min_crop_row_angle))
        min_rad_angle = self.compas_dregree_angle_to_hough_rad(self.min_crop_row_angle)
        max_rad_angle = self.compas_dregree_angle_to_hough_rad(self.max_crop_row_angle)
        tested_angles = np.linspace(min_rad_angle, max_rad_angle, number_of_angles)

        # scipy's implementation er anvendt, da denne for nu er hurtigere
        # self.h, self.theta, self.d = hough_transform_grayscale.hough_line(self.gray, theta=tested_angles)
        h, tile_img_data.theta, tile_img_data.d = hough_line(tile_img_data.gray, theta=tested_angles)
        h = h.astype(np.float32)

        h = self.normalize_array(h)
        self.write_image_to_file("33_hough_image.png", 255 * h, tile)

        h = self.blur_image(h)
        h = self.normalize_array(h)
        self.write_image_to_file("34_hough_image_blurred.png", 255 * h, tile)

        h = self.apply_top_hat(h, tile)

        tile_img_data.h = self.normalize_array(h)

        self.write_image_to_file("35_hough_image_tophat.png", 255 * h, tile)

    def normalize_array(self, arr):
        _max = cv2.minMaxLoc(arr)[1]
        if _max > 0:
            arr = arr / _max
        else:
            # This is implemented to stop the padding tiles from being 0
            # and therefore throwing an error when using np.log, as log(0) is undefined.
            arr = arr + 10e-10
        return arr

    def determine_dominant_direction(self, tile, tile_img_data):
        baseline_fitter = Baseline(tile_img_data.theta * 180 / np.pi, check_finite=False)

        # There are 4 different ways to determine the dominant row, as seen below.
        tile_img_data.direction_response = np.sum(np.square(tile_img_data.h), axis=0)
        tile_img_data.log_direc = np.log(tile_img_data.direction_response)
        tile_img_data.log_direc_baseline = (
            np.log(tile_img_data.direction_response)
            - baseline_fitter.mor(np.log(tile_img_data.direction_response), half_window=30)[0]
        )
        tile_img_data.direc_baseline = (
            tile_img_data.direction_response - baseline_fitter.mor(tile_img_data.direction_response, half_window=30)[0]
        )

        tile_img_data.direction_with_most_energy_idx = np.argmax(tile_img_data.direc_baseline)
        tile.direction = tile_img_data.theta[tile_img_data.direction_with_most_energy_idx]

        self.plot_direction_energies(tile, tile_img_data)

        """
        # Plot the direction response and normalized direction response
        max = np.max(log_direc_baseline)
        plt.figure(figsize=(16, 9))
        plt.plot(tile.theta*180/np.pi,
                 log_direc_baseline,
                 color='orange')
        if max != 0:
            plt.plot(tile.theta*180/np.pi,
                     log_direc_baseline/max,
                     color='blue')
        self.write_plot_to_file("37_direction_energies_normalized.png", tile)
        plt.close()
        """

    def plot_direction_energies(self, tile, tile_img_data):
        plt.figure(figsize=(16, 9))
        self.plot_direction_response_and_maximum(
            tile_img_data, tile_img_data.log_direc, "blue", "log of direction response"
        )
        self.plot_direction_response_and_maximum(
            tile_img_data, tile_img_data.direc_baseline, "green", "direction response - baseline"
        )
        self.plot_direction_response_and_maximum(
            tile_img_data, tile_img_data.log_direc_baseline, "orange", "log of direction response - baseline"
        )
        self.plot_direction_response_and_maximum(
            tile_img_data, tile_img_data.direction_response, "red", "direction response"
        )
        plt.legend()
        self.write_plot_to_file("36_direction_energies.png", tile)
        plt.close()

    def plot_direction_response_and_maximum(self, tile_img_data, direction_response, color, label):
        plt.plot(tile_img_data.theta * 180 / np.pi, direction_response, color=color, label=label)
        plt.axvline(x=tile_img_data.theta[np.argmax(direction_response)] * 180 / np.pi, color=color, linestyle="dashed")

    def determine_offsets_of_crop_rows(self, tile, tile_img_data):
        tile_img_data.signal = tile_img_data.h[:, tile_img_data.direction_with_most_energy_idx]

        tile_img_data.peaks, _ = find_peaks(
            tile_img_data.signal, distance=tile.expected_crop_row_distance / 2, prominence=0.01
        )

        self.plot_row_offset(tile, tile_img_data)

        self.plot_row_offset_with_peaks(tile, tile_img_data)

    def plot_row_offset(self, tile, tile_img_data):
        plt.figure(figsize=(16, 9))
        plt.plot(tile_img_data.signal, color="blue")
        self.write_plot_to_file("38_row_offsets.png", tile)
        plt.close()

    def plot_row_offset_with_peaks(self, tile, tile_img_data):
        plt.figure(figsize=(16, 9))
        plt.plot(tile_img_data.signal)
        plt.plot(tile_img_data.peaks, tile_img_data.signal[tile_img_data.peaks], "x")
        plt.plot(np.zeros_like(tile_img_data.signal), "--", color="gray")
        self.write_plot_to_file("39_row_offsets_with_detected_peaks.png", tile)
        plt.close()

    def determine_line_ends_of_crop_rows(self, tile, tile_img_data):
        tile.vegetation_lines = []
        prev_peak_dist = 0

        for peak_idx in tile_img_data.peaks:
            dist = tile_img_data.d[peak_idx]

            angle = tile.direction

            self.fill_in_gaps_in_detected_crop_rows(dist, prev_peak_dist, angle, tile, tile_img_data)
            line_ends = self.get_line_ends_within_image(dist, angle, tile_img_data.veg_img_constant)

            prev_peak_dist = dist
            tile.vegetation_lines.append(line_ends)

        vegetation_lines = []
        for line_ends in tile.vegetation_lines:
            if len(line_ends) == 2:
                vegetation_lines.append(line_ends)
        tile.vegetation_lines = vegetation_lines

    def draw_detected_crop_rows_on_input_image_and_segmented_image(self, tile, tile_img_data):
        for line_ends in tile.vegetation_lines:
            try:
                self.draw_crop_row(tile_img_data.veg_img, line_ends)
                self.draw_crop_row(tile_img_data.gray_inverse, line_ends)
            except Exception as e:
                print(e)
                ic(line_ends)

        self.write_image_to_file("40_detected_crop_rows.png", tile_img_data.veg_img, tile)
        self.write_image_to_file("45_detected_crop_rows_on_segmented_image.png", tile_img_data.gray_inverse, tile)

    def draw_crop_row(self, image, line_ends):
        cv2.line(image, (line_ends[0][0], line_ends[0][1]), (line_ends[1][0], line_ends[1][1]), (0, 0, 255), 1)

    def add_boundary_and_number_to_tile(self, tile, tile_img_data):
        if tile.tile_boundary:
            cv2.line(tile_img_data.veg_img, (0, 0), (tile_img_data.veg_img.shape[1] - 1, 0), (0, 0, 255), 1)
            cv2.line(
                tile_img_data.veg_img,
                (0, tile_img_data.veg_img.shape[0] - 1),
                (tile_img_data.veg_img.shape[1] - 1, tile_img_data.veg_img.shape[0] - 1),
                (0, 0, 255),
                1,
            )
            cv2.line(tile_img_data.veg_img, (0, 0), (0, tile_img_data.veg_img.shape[0] - 1), (0, 0, 255), 1)
            cv2.line(
                tile_img_data.veg_img,
                (tile_img_data.veg_img.shape[1] - 1, 0),
                (tile_img_data.veg_img.shape[1] - 1, tile_img_data.veg_img.shape[0] - 1),
                (0, 0, 255),
                1,
            )
            cv2.putText(
                tile_img_data.veg_img,
                f"{tile.tile_number}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def fill_in_gaps_in_detected_crop_rows(self, dist, prev_peak_dist, angle, tile, tile_img_data):
        if prev_peak_dist != 0:
            while self.distance_between_two_peaks_is_larger_than_expected(dist, prev_peak_dist, tile):
                prev_peak_dist += tile.expected_crop_row_distance
                line_ends = self.get_line_ends_within_image(prev_peak_dist, angle, tile_img_data.veg_img_constant)
                tile.filler_rows.append([line_ends, len(tile.vegetation_lines), tile.tile_number])
                tile.vegetation_lines.append(line_ends)

    def distance_between_two_peaks_is_larger_than_expected(self, dist, prev_peak_dist, tile):
        return dist - prev_peak_dist > 2 * tile.expected_crop_row_distance

    def get_line_ends_within_image(self, dist, angle, img):
        x_val_range = np.array((0, img.shape[1]))
        y_val_range = np.array((0, img.shape[0]))
        # x * cos(t) + y * sin(t) = r
        y0, y1 = (dist - x_val_range * np.cos(angle)) / np.sin(angle)
        x0, x1 = (dist - y_val_range * np.sin(angle)) / np.cos(angle)
        line_ends = []
        if int(y0) >= 0 and int(y0) <= img.shape[0]:
            line_ends.append([0, int(y0)])
        if int(x0) >= 0 and int(x0) <= img.shape[1]:
            line_ends.append([int(x0), 0])
        if int(x1) >= 0 and int(x1) <= img.shape[1]:
            line_ends.append([int(x1), img.shape[0]])
        if int(y1) >= 0 and int(y1) <= img.shape[0]:
            line_ends.append([img.shape[0], int(y1)])

        try:
            if line_ends[0][0] > line_ends[1][0]:
                line_ends = [line_ends[1], line_ends[0]]
        except IndexError:
            pass

        return line_ends

    def measure_vegetation_coverage_in_crop_row(self, tile, tile_img_data):
        # 1. Blur image with a uniform kernel
        # Approx distance between crop rows is 16 pixels.
        # I would prefer to have a kernel size that is not divisible by two.
        vegetation_map = cv2.blur(tile_img_data.gray.astype(np.uint8), (10, 10))
        self.write_image_to_file("60_vegetation_map.png", vegetation_map, tile)

        # 2. Sample pixel values along each crop row
        #    - cv2.remap
        # Hardcoded from earlier run of the algorithm.
        # missing_plants_image = cv2.cvtColor(vegetation_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        missing_plants_image = tile_img_data.veg_img
        df_missing_vegetation_list = []

        DF_combined = pd.DataFrame({"tile": [], "row": [], "x": [], "y": [], "vegetation": []})

        for counter, crop_row in enumerate(tile.vegetation_lines):
            try:
                self.find_vegetation_in_crop_row(
                    tile, vegetation_map, missing_plants_image, df_missing_vegetation_list, counter, crop_row
                )
            except Exception as e:
                print(e)
                traceback.print_exc()
                print("measure_vegetation_coverage_in_crop_row: ")
                print("tile: ", tile.tile_number)

        if df_missing_vegetation_list:
            DF_combined = pd.concat(df_missing_vegetation_list)

        if tile.generate_debug_images:
            filename = self.get_debug_output_filepath("68_vegetation_samples.csv", tile)
            DF_combined.to_csv(filename, index=False)
        self.write_image_to_file("67_missing_plants_in_crop_line.png", missing_plants_image, tile)

        # 3. Export to a csv file, include the following information
        #    - row number and offset
        #    - pixel coordinates
        #    - vegetation coverage

    def find_vegetation_in_crop_row(
        self, tile, vegetation_map, missing_plants_image, df_missing_vegetation_list, counter, crop_row
    ):
        x_sample_coords, y_sample_coords = self.calculate_x_and_y_sample_cords_along_crop_row(tile, crop_row)

        DF = self.create_data_structure_containing_crop_row(
            tile, vegetation_map, counter, x_sample_coords, y_sample_coords
        )
        df_missing_vegetation_list.append(DF)

        self.plot_points_without_vegetation_on_crop_row(tile, missing_plants_image, DF)

    def plot_points_without_vegetation_on_crop_row(self, tile, missing_plants_image, DF):
        threshold_vegetation = 60
        missing_plants = DF[DF["vegetation"] < threshold_vegetation]
        for _index, location in missing_plants.iterrows():
            cv2.circle(
                missing_plants_image,
                (
                    int(location["x"] - tile.size[0] * tile.tile_position[1]),
                    int(location["y"] - tile.size[1] * tile.tile_position[0]),
                ),
                2,
                (255, 255, 0),
                -1,
            )

    def create_data_structure_containing_crop_row(
        self, tile, vegetation_map, counter, x_sample_coords, y_sample_coords
    ):
        vegetation_samples = cv2.remap(
            vegetation_map, x_sample_coords.astype(np.float32), y_sample_coords.astype(np.float32), cv2.INTER_LINEAR
        )

        DF = pd.DataFrame(
            {
                "tile": tile.tile_number,
                "row": counter,
                "x": x_sample_coords + tile.size[0] * tile.tile_position[1],
                "y": y_sample_coords + tile.size[1] * tile.tile_position[0],
                "vegetation": vegetation_samples.transpose()[0],
            }
        )

        return DF

    def calculate_x_and_y_sample_cords_along_crop_row(self, tile, crop_row):
        angle = tile.direction
        # Determine sample locations along the crop row
        start_point = (crop_row[0][0], crop_row[0][1])

        end_point = (crop_row[1][0], crop_row[1][1])
        distance = np.linalg.norm(np.asarray(start_point) - np.asarray(end_point))

        distance_between_samples = 1
        n_samples = np.ceil(0.0001 + distance / distance_between_samples)
        assert n_samples > 0, "n_samples is less than 0"

        x_close_to_end = start_point[0] + distance * np.sin(angle)
        y_close_to_end = start_point[1] + distance * np.cos(angle) * (-1)

        # In some cases the given angle points directly away from the end point, instead of
        # point towards the end point from the starting point. In that case, reverse the direction.
        if np.abs(x_close_to_end - end_point[0]) + np.abs(y_close_to_end - end_point[1]) > 5:
            angle = angle + np.pi

        x_sample_coords = start_point[0] + range(0, int(n_samples)) * np.sin(angle) * (1)
        y_sample_coords = start_point[1] + range(0, int(n_samples)) * np.cos(angle) * (-1)
        return x_sample_coords, y_sample_coords

    def convert_segmented_image_to_binary(self, tile, tile_img_data):
        binary = tile_img_data.segmented_img
        for i in range(0, binary.shape[0]):
            for j in range(0, binary.shape[1]):
                if binary[i, j] < tile.threshold_level:
                    binary[i, j] = 255
                else:
                    binary[i, j] = 0
        tile_img_data.gray = binary
        tile_img_data.gray_inverse = 255 - binary

    def load_tile_with_data_needed_for_crop_row_detection(self, tile):
        tile.generate_debug_images = self.generate_debug_images
        tile.tile_boundary = self.tile_boundary
        tile.threshold_level = self.threshold_level
        # In gimp I have measured the crop row distance to be around 20 px.
        # however I get the best results when this value is set to 30.
        tile.expected_crop_row_distance = self.expected_crop_row_distance

        tile.gray = None
        tile.gray_inverse = None

        # Data for the crop row detector
        # Hough transform and directions
        tile.h = None
        tile.theta = None
        tile.d = None
        tile.direction_with_most_energy_idx = None
        tile.direction = None
        tile.peaks = None

        # Save the endpoints of the detected crop rows
        tile.vegetation_lines = []
        # List containing the lacking rows
        tile.filler_rows = []

    def combine_segmented_and_original_tile(self, tile_segmented, tile_plot):
        tile_segmented_img = tile_segmented.read_img()
        tile_plot_img = tile_plot.read_img()

        assert tile_segmented_img.shape[0] == 1, "The segmented image has more then one color chanal."
        assert tile_plot.ulc == tile_segmented.ulc, "The two tiles are not the same location."

        tile_img_data = tile_data_holder()

        tile_img_data.segmented_img = tile_segmented_img.reshape(
            tile_segmented_img.shape[1], tile_segmented_img.shape[2]
        )

        if tile_plot_img.shape[0] == 1:
            # tile_plot.img = tile_plot.img.reshape(tile_plot.img.shape[1], tile_plot.img.shape[2]).copy()
            temp_tile_plot = tile_plot_img.reshape(tile_plot_img.shape[1], tile_plot_img.shape[2])
        else:
            temp_tile_plot = tile_plot_img
        # tile_segmented.img = tile_plot.img.copy()
        # tile_segmented.img_constant = tile_plot.img.copy()
        tile_img_data.veg_img = temp_tile_plot
        tile_img_data.veg_img_constant = temp_tile_plot
        return tile_img_data

    def main(self, tiles_segmented, tiles_plot, args):
        tile_pairs = list(zip(tiles_segmented, tiles_plot, strict=False))

        start = time.time()
        total_results = []

        if self.run_parallel:
            total_results = process_map(self.detect_crop_rows, tile_pairs, chunksize=1, max_workers=self.max_workers)
        else:
            total_results = list(tqdm(map(self.detect_crop_rows, tile_pairs), total=len(tile_pairs)))

        print("Time to run all tiles: ", time.time() - start)

        # tiles_segmented = list(total_results.copy())
        total_results = list(total_results)

        self.create_csv_of_row_information(total_results)
        self.create_csv_of_row_information_global(total_results)
        self.vegetation_row_to_csv(total_results)
        self.save_statistics(args, total_results)

    def detect_crop_rows(self, tile_pairs):
        self.load_tile_with_data_needed_for_crop_row_detection(tile_pairs[0])
        tile_img_data = self.combine_segmented_and_original_tile(tile_pairs[0], tile_pairs[1])

        try:
            self.convert_segmented_image_to_binary(tile_pairs[0], tile_img_data)
        except Exception as a:
            ic(a)
        try:
            self.apply_hough_lines(tile_pairs[0], tile_img_data)
        except Exception as b:
            ic(b)
        try:
            self.determine_dominant_direction(tile_pairs[0], tile_img_data)
        except Exception as c:
            ic(c)
        try:
            self.determine_offsets_of_crop_rows(tile_pairs[0], tile_img_data)
        except Exception as d:
            ic(d)
        try:
            self.determine_line_ends_of_crop_rows(tile_pairs[0], tile_img_data)
        except Exception as e:
            ic(e)
        try:
            self.draw_detected_crop_rows_on_input_image_and_segmented_image(tile_pairs[0], tile_img_data)
        except Exception as f:
            ic(f)
        try:
            self.measure_vegetation_coverage_in_crop_row(tile_pairs[0], tile_img_data)
        except Exception as g:
            ic(g)
        try:
            self.add_boundary_and_number_to_tile(tile_pairs[0], tile_img_data)
        except Exception as h:
            ic(h)
        try:
            tile_pairs[0].save_tile(tile_img_data.veg_img)
        except Exception as i:
            ic(i)
        return tile_pairs[0]

    def create_csv_of_row_information(self, tiles_segmented):
        row_information = []

        for tile in tiles_segmented:
            if tile.direction < 0:
                tile.direction = np.pi + tile.direction
            for row_number, row in enumerate(tile.vegetation_lines):
                row_information.append(
                    [
                        tile.tile_number,
                        tile.tile_position[0],
                        tile.tile_position[1],
                        tile.direction,
                        row_number,
                        row[0][0],
                        row[0][1],
                        row[1][0],
                        row[1][1],
                    ]
                )

        DF_row_information = pd.DataFrame(
            row_information,
            columns=["tile", "x_position", "y_position", "angle", "row", "x_start", "y_start", "x_end", "y_end"],
        )

        csv_path = tiles_segmented[0].output_tile_location
        DF_row_information.to_csv(csv_path + "/row_information.csv")

    def create_csv_of_row_information_global(self, tiles_segmented):
        row_information = []

        for tile in tiles_segmented:
            if tile.direction < 0:
                tile.direction = np.pi + tile.direction
            for row_number, row in enumerate(tile.vegetation_lines):
                row_information.append(
                    [
                        tile.tile_number,
                        tile.tile_position[0],
                        tile.tile_position[1],
                        tile.direction,
                        row_number,
                        (row[0][0] - tile.size[0] * tile.tile_position[1]) * tile.resolution[0] + tile.ulc_global[1],
                        (row[0][1] - tile.size[1] * tile.tile_position[0]) * tile.resolution[1] + tile.ulc_global[0],
                        (row[1][0] - tile.size[0] * tile.tile_position[1]) * tile.resolution[0] + tile.ulc_global[1],
                        (row[1][1] - tile.size[1] * tile.tile_position[0]) * tile.resolution[1] + tile.ulc_global[0],
                    ]
                )

        DF_row_information = pd.DataFrame(
            row_information,
            columns=["tile", "x_position", "y_position", "angle", "row", "x_start", "y_start", "x_end", "y_end"],
        )

        csv_path = tiles_segmented[0].output_tile_location
        DF_row_information.to_csv(csv_path + "/row_information_global.csv")

    def vegetation_row_to_csv(self, tiles_segmented):
        DF_vegetation_rows = pd.DataFrame(columns=["tile", "row", "x", "y", "vegetation"])
        csv_path = tiles_segmented[0].output_tile_location + "/points_in_rows.csv"
        DF_vegetation_rows.to_csv(csv_path, index=False)
        print("tile_number: ", tiles_segmented[0].tile_number)
        print("tile_position: ", tiles_segmented[0].ulc_global)
        print("tile_transform: ", tiles_segmented[0].transform)
        print("tile_resolution: ", tiles_segmented[0].resolution)

        for tile in tiles_segmented:
            filename = self.get_debug_output_filepath("68_vegetation_samples.csv", tile)
            DF_vegetation_rows = pd.read_csv(filename)
            DF_vegetation_rows["x"] = (
                DF_vegetation_rows["x"] - tile.size[0] * tile.tile_position[1]
            ) * tile.resolution[0] + tile.ulc_global[1]
            DF_vegetation_rows["y"] = (
                -(DF_vegetation_rows["y"] - tile.size[1] * tile.tile_position[0]) * tile.resolution[1]
                + tile.ulc_global[0]
            )
            DF_vegetation_rows.to_csv(csv_path, mode="a", index=False, header=False)

    def save_statistics(self, args, tiles_segmented):
        statistics_path = tiles_segmented[0].output_tile_location + "/statistics"
        self.ensure_parent_directory_exist(statistics_path + "/output_file.txt")

        print(f'Writing statistics to the folder "{ statistics_path }"')

        with open(statistics_path + "/output_file.txt", "w") as f:
            f.write("Input parameters:\n")
            f.write(f" - Segmented Orthomosaic: {args.segmented_orthomosaic}\n")
            f.write(f" - Orthomosaic: {args.orthomosaic}\n")
            f.write(f" - Tile sizes: {args.tile_size}\n")
            f.write(f" - Output tile location: {args.output_tile_location}\n")
            f.write(f" - Generated debug images: {args.generate_debug_images}\n")
            f.write(f" - Tile boundary: {args.tile_boundary}\n")
            f.write(f" - Ecpected crop row distance: {args.expected_crop_row_distance}\n")
            f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")
            f.write("\n\nOutput from run\n")
            f.write(f" - Number of tiles: {len(tiles_segmented)}\n")
