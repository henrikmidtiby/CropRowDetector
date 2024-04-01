import cv2
import numpy as np
import pandas as pd
from path import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from icecream import ic
from pybaselines import Baseline

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from tqdm import tqdm
import os

import hough_transform_grayscale
from skimage.transform import hough_line

import time


# python3 crop_row_detector.py 2023-05-01_Alm_rajgraes_cleopatra_jsj_imput_image --generate_debug_images True --debug_image_folder output 

class Tile:
    def __init__(self, start_point, position, height, width):
        # Data for the tile
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.ulc_global = []
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]

        # Data for the detected crop rows
        # Tile
        
        self.tile_number = None
        self.img = None
        self.gray = None

        # Hough transform and directions
        self.h = None
        self.theta = None
        self.d = None
        self.direction_with_most_energy_idx = None
        self.direction = None
        self.peaks = None
        

        self.vegetation = []
        # Save the endpoints of the detected crop rows
        self.vegetation_lines = []
        # List containing the lacking rows
        self.filler_rows = []

        self.threshold_level = 10
        self.generate_debug_images = None
        self.tile_boundry = None

        self.output_tile_location = None

        # In gimp I have measured the crop row distance to be around 20 px.
        # however I get the best results when this value is set to 30.
        self.expected_crop_row_distance = 20 # 30


def rasterio_opencv2(image):
    if image.shape[0] >= 3:  # might include alpha channel
        false_color_img = image.transpose(1, 2, 0)
        separate_colors = cv2.split(false_color_img)
        return cv2.merge([separate_colors[2],
                          separate_colors[1],
                          separate_colors[0]])
    else:
        return image


def read_tile(orthomosaic, tile):
    with rasterio.open(orthomosaic) as src:
        window = Window.from_slices((tile.ulc[0], tile.lrc[0]),
                                    (tile.ulc[1], tile.lrc[1]))
        im = src.read(window=window)
    return rasterio_opencv2(im)

class crop_row_detector:
    #def __init__(self):
        
        # This class is just a crop row detctor in form of a collection of functions, 
        # all of the information is stored in the information class Tile.
        
        

        



    def ensure_parent_directory_exist(self, path):
        temp_path = Path(path).parent
        if not temp_path.exists():
            temp_path.mkdir()

    def write_image_to_file(self, output_path, img, tile): 
        if tile.generate_debug_images:
            path = self.get_debug_output_filepath(output_path, tile)
            self.ensure_parent_directory_exist(path)
            cv2.imwrite(path, img)

    def get_debug_output_filepath(self, output_path, tile):
        return tile.output_tile_location + "/debug_images/" + f'{tile.tile_number}' + "/" + output_path

    def write_plot_to_file(self, output_path, tile):
        if tile.generate_debug_images:
            path = self.get_debug_output_filepath(output_path, tile)
            self.ensure_parent_directory_exist(path)
            plt.savefig(path, dpi=300)

    def apply_top_hat(self, h, tile):
        # column filter with the distance between 2 rows
        filterSize = (1, int(tile.expected_crop_row_distance)) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  
                                        filterSize) 
        # Applying the Top-Hat operation 
        h = cv2.morphologyEx(h,  
                                    cv2.MORPH_TOPHAT, 
                                    kernel)
        return h

    def apply_hough_lines(self, tile):
        # Apply the hough transform
        number_of_angles = 8*360
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
        
        #t1 = time.time()
        #scipy's implementation er anvendt, da denne for nu er hurtigere
        #self.h, self.theta, self.d = hough_transform_grayscale.hough_line(self.gray, theta=tested_angles)
        #t2 = time.time()
        h, tile.theta, tile.d = hough_line(tile.gray, theta=tested_angles)
        
        #print("Time to run hough transform: ", t2 - t1)
        #print("Time to run hough transform: ", time.time() - t2)
        h = h.astype(np.float32)
        h = self.divide_by_max_in_array(h)
        self.write_image_to_file("33_hough_image.png", 255 * h, tile)

        # Blur image using a 5 x 1 average filter
        kernel = np.ones((5,1), np.float32) / 5
        h = cv2.filter2D(h, -1, kernel)
        h = self.divide_by_max_in_array(h)
        self.write_image_to_file("34_hough_image_blurred.png", 255 * h, tile)

        h = self.apply_top_hat(h, tile)
        
        tile.h = self.divide_by_max_in_array(h)

        self.write_image_to_file("35_hough_image_tophat.png", 255 * h, tile)

    def divide_by_max_in_array(self, arr):
        temp = cv2.minMaxLoc(arr)[1]
        if temp > 0:
            arr = arr/temp
        return arr

    def determine_dominant_row(self, tile):
        # Determine the dominant row direction
        direction_response = np.sum(np.square(tile.h), axis=0)
        baseline_fitter = Baseline(tile.theta*180/np.pi, check_finite=False)

        print("h: ", tile.h)

        print("subtract problem: ", np.log(direction_response), baseline_fitter.mor(np.log(direction_response), half_window=30)[0])
        
        # Normalize the direction response
        Direc_energi = np.log(direction_response) - baseline_fitter.mor(np.log(direction_response), half_window=30)[0]
        max = np.max(Direc_energi)

        


        # the direction with the most energi is dicided from sum of the squrare of the hough transform
        # it is possible to subtrack the baseline, but this does not always provide a better result.
        tile.direction_with_most_energy_idx = np.argmax(direction_response)#Direc_energi)
        tile.direction = tile.theta[tile.direction_with_most_energy_idx]
        
        # Plot the direction response and normalized direction response
        plt.figure(figsize=(16, 9))
        plt.plot(tile.theta*180/np.pi, np.log(direction_response), color='blue')
        self.write_plot_to_file("36_direction_energies.png", tile)
        plt.plot(tile.theta*180/np.pi, Direc_energi, color='orange')
        self.write_plot_to_file("36_direction_energies_2.png", tile)
        plt.close()

        plt.figure(figsize=(16, 9))
        plt.plot(tile.theta*180/np.pi, 
                 Direc_energi, 
                 color='orange')
        if max != 0:
            plt.plot(tile.theta*180/np.pi, 
                     Direc_energi/max,
                     color='blue')
        self.write_plot_to_file("37_direction_energies_normalized.png", tile)
        plt.close()

    def plot_counts_in_hough_accumulator_with_direction(self, tile):
        plt.figure(figsize=(16, 9))
        plt.plot(tile.h[:, tile.direction_with_most_energy_idx])
        self.write_plot_to_file("38_row_offsets.png", tile)
        plt.close()

    def determine_and_plot_offsets_of_crop_rows_with_direction(self, tile):
        signal = tile.h[:, tile.direction_with_most_energy_idx]

        
        peaks, _ = find_peaks(signal, distance=tile.expected_crop_row_distance / 2)
        tile.peaks = peaks
        plt.figure(figsize=(16, 9))
        plt.plot(signal)
        plt.plot(peaks, signal[peaks], "x")
        plt.plot(np.zeros_like(signal), "--", color="gray")
        self.write_plot_to_file("39_signal_with_detected_peaks.png", tile)
        plt.close()

    def draw_detected_crop_rows_on_input_image(self, tile):
        tile.vegetation_lines = []
        # Draw detected crop rows on the input image
        origin = np.array((0, tile.img.shape[1])) 
        prev_peak_dist = 0
        for peak_idx in tile.peaks:
            dist = tile.d[peak_idx]
            angle = tile.direction

            self.fill_in_gaps_in_detected_crop_rows(dist, prev_peak_dist, angle, tile)

            temp = self.get_line_ends_within_image(dist, angle, tile.img)
            try:
                cv2.line(tile.img, (temp[0][0], temp[0][1]), 
                         (temp[1][0], temp[1][1]), (0, 0, 255), 1)
            except Exception as e:
                print(e)
                ic(temp)
            prev_peak_dist = dist
            tile.vegetation_lines.append(temp)
        if tile.tile_boundry:
            self.add_boundary_and_number_to_tile(tile)
        self.write_image_to_file("40_detected_crop_rows.png", tile.img, tile)

    def add_boundary_and_number_to_tile(self, tile):
        cv2.line(tile.img, (0, 0), (tile.img.shape[1]-1, 0), (0, 0, 255), 1)
        cv2.line(tile.img, (0, tile.img.shape[0]-1), (tile.img.shape[1]-1, tile.img.shape[0]-1), (0, 0, 255), 1)
        cv2.line(tile.img, (0, 0), (0, tile.img.shape[0]-1), (0, 0, 255), 1)
        cv2.line(tile.img, (tile.img.shape[1]-1, 0), (tile.img.shape[1]-1, tile.img.shape[0]-1), (0, 0, 255), 1)
        cv2.putText(tile.img, f'{tile.tile_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def fill_in_gaps_in_detected_crop_rows(self, dist, prev_peak_dist, angle, tile):
        # If distance between two rows is larger than twice the expected row distance,
        # Then fill in the gap with lines.
        if prev_peak_dist != 0 and dist - prev_peak_dist > 2 * tile.expected_crop_row_distance:
            while prev_peak_dist + tile.expected_crop_row_distance < dist-tile.expected_crop_row_distance:
                prev_peak_dist += tile.expected_crop_row_distance
                temp = self.get_line_ends_within_image(prev_peak_dist, angle, tile.img)
                cv2.line(tile.img, (temp[0][0], temp[0][1]), 
                            (temp[1][0], temp[1][1]), (0, 0, 255), 1)
                tile.filler_rows.append([temp, len(tile.vegetation_lines), tile.tile_number])
                tile.vegetation_lines.append(temp)
                

    def get_line_ends_within_image(self, dist, angle, img):
        x_val_range = np.array((0, img.shape[1]))
        y_val_range = np.array((0, img.shape[0]))
        # x * cos(t) + y * sin(t) = r
        y0, y1 = (dist - x_val_range * np.cos(angle)) / np.sin(angle)
        x0, x1 = (dist - y_val_range * np.sin(angle)) / np.cos(angle)
        temp = []
        #print("y0: ", y0, "y1: ", y1, "x0: ", x0, "x1: ", x1)
        #print("y0: ", int(y0), "y1: ", int(y1), "x0: ", int(x0), "x1: ", int(x1))
        if int(y0) >= -1 and int(y0) <= img.shape[0]:
            temp.append([0, int(y0)])
        if int(y1) >= -1 and int(y1) <= img.shape[0]:
            temp.append([img.shape[0], int(y1)])
        if int(x0) >= -1 and int(x0) <= img.shape[1]:
            temp.append([int(x0), 0])
        if int(x1) >= -1 and int(x1) <= img.shape[1]:
            temp.append([int(x1), img.shape[0]])
        #print("temp: ", temp)
        return temp     

    def measure_vegetation_coverage_in_crop_row(self, tile):
        # 1. Blur image with a uniform kernel
        # Approx distance between crop rows is 16 pixels.
        # I would prefer to have a kernel size that is not divisible by two.
        temp = tile.gray.astype(np.uint8)
        vegetation_map = cv2.blur(temp, (10, 10))
        self.write_image_to_file("60_vegetation_map.png", vegetation_map, tile)

        # 2. Sample pixel values along each crop row
        #    - cv2.remap
        # Hardcoded from earlier run of the algorithm.
        # missing_plants_image = cv2.cvtColor(vegetation_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        missing_plants_image = tile.img
        df_missing_vegetation_list = []

        DF_combined = pd.DataFrame({'tile': [],
                                'row': [], 
                                'x': [], 
                                'y': [], 
                                'vegetation': []})

        for counter, crop_row in enumerate(tile.vegetation_lines):
            try:
                angle = tile.direction
                # Determine sample locations along the crop row
                start_point = (crop_row[0][0], crop_row[0][1])
                distance_between_samples = 1
                end_point = (crop_row[1][0], crop_row[1][1])
                distance = np.linalg.norm(np.asarray(start_point) - np.asarray(end_point))
                n_samples = np.ceil(distance / distance_between_samples)

                x_close_to_end = start_point[0] + distance * np.sin(angle)
                y_close_to_end = start_point[1] + distance * np.cos(angle) * (-1)
                
                # In some cases the given angle points directly away from the end point, instead of
                # point towards the end point from the starting point. In that case, reverse the direction.
                if np.abs(x_close_to_end - end_point[0]) + np.abs(y_close_to_end - end_point[1]) > 5:
                    angle = angle + np.pi

                x_sample_coords = start_point[0] + range(0, int(n_samples)) * np.sin(angle) * (1)
                y_sample_coords = start_point[1] + range(0, int(n_samples)) * np.cos(angle) * (-1)

                vegetation_samples = cv2.remap(vegetation_map, 
                                            x_sample_coords.astype(np.float32), 
                                            y_sample_coords.astype(np.float32), 
                                            cv2.INTER_LINEAR)

                DF = pd.DataFrame({'tile': tile.tile_number,
                                'row': counter, 
                                'x': x_sample_coords + tile.size[0]*tile.tile_position[1], 
                                'y': y_sample_coords + tile.size[1]*tile.tile_position[0], 
                                'vegetation': vegetation_samples.transpose()[0]})
                df_missing_vegetation_list.append(DF)
                

                threshold_vegetation = 60
                missing_plants = DF[DF['vegetation'] < threshold_vegetation]
                for index, location in missing_plants.iterrows():
                    cv2.circle(missing_plants_image, 
                            (int(location['x'] - tile.size[0]*tile.tile_position[1]), 
                             int(location['y'] - tile.size[1]*tile.tile_position[0])), 
                            2, 
                            (255, 255, 0), 
                            -1)
            except Exception as e:
                print(e)
        
        #print("df_missing_vegetation_list: ", df_missing_vegetation_list)

        #filename = self.date_time + "/" + "64_vegetation_samples.csv"
        filename = tile.output_tile_location + "/debug_images/" + f'{tile.tile_number}' + "/" + "68_vegetation_samples.csv"
        if df_missing_vegetation_list:
            #print("df_missing_vegetation_list: ", df_missing_vegetation_list)
            DF_combined = pd.concat(df_missing_vegetation_list)
            
        #print("Df: ", DF_combined, "\n\n")
        #print("Df: ", DF_combined.shape, "\n\n")
        DF_combined.to_csv(filename)
        self.write_image_to_file("67_missing_plants_in_crop_line.png", missing_plants_image, tile)
    
        # 3. Export to a csv file, include the following information
        #    - row number and offset
        #    - pixel coordinates
        #    - vegetation coverage   


    def draw_detected_crop_rows_on_segmented_image(self, tile):
        segmented_annotated = tile.gray.copy()
        # Draw detected crop rows on the segmented image
        origin = np.array((0, segmented_annotated.shape[1]))
        segmented_annotated = 255 - segmented_annotated
        for peak_idx in tile.peaks:
            dist = tile.d[peak_idx]
            angle = tile.direction
            temp = self.get_line_ends_within_image(dist, angle, tile.img)
            try:
                self.draw_crop_row(segmented_annotated, temp)
            except Exception as e:
                print(e)
                ic(temp)
        self.segmented_annotated = segmented_annotated
        self.write_image_to_file("45_detected_crop_rows_on_segmented_image.png", segmented_annotated, tile)

    def draw_crop_row(self, segmented_annotated, temp):
        cv2.line(segmented_annotated, 
                            (temp[0][0], temp[0][1]), 
                            (temp[1][0], temp[1][1]), 
                            (0, 0, 255), 1)

    # Dette burde aldrig køres, da der altid burde være et segmenteret billede
    def convert_to_grayscale(self, tile):
        HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        HSV_gray = HSV[:,:,2].copy()
        #self.write_image_to_file("02_hsv.png", HSV_gray)
        for i in range(0, HSV_gray.shape[0]):
            for j in range(0, HSV_gray.shape[1]):
                if HSV[i,j,0] > 45 and HSV[i,j,0] < 65:
                    HSV_gray[i,j] = 255-HSV_gray[i,j]
                else:
                    HSV_gray[i,j] = 0
        #self.write_image_to_file("03_hsv.png", HSV_gray)
        tile.gray = HSV_gray

    def gray_reduce(self, tile):
        gray_temp = tile.gray.copy()
        for i in range(0, gray_temp.shape[0]):
            for j in range(0, gray_temp.shape[1]):
                if gray_temp[i,j] < tile.threshold_level:
                    gray_temp[i,j] = 255
                else:
                    gray_temp[i,j] = 0
        tile.gray = gray_temp


    def main(self, segmented_img, input_orthomosaic, tile):
        # run row detection on tile
        if segmented_img.shape[0] == 1:
            tile.gray = segmented_img.reshape(segmented_img.shape[1], segmented_img.shape[2])
            if input_orthomosaic is not None:
                tile.img = input_orthomosaic
            else: 
                tile.img = segmented_img.reshape(segmented_img.shape[1], segmented_img.shape[2])
            self.gray_reduce(tile)
        else:
            tile.img = segmented_img
            self.convert_to_grayscale(tile)
        
        self.apply_hough_lines(tile)
        self.determine_dominant_row(tile)
        self.plot_counts_in_hough_accumulator_with_direction(tile)
        self.determine_and_plot_offsets_of_crop_rows_with_direction(tile)
        self.draw_detected_crop_rows_on_input_image(tile)
        if tile.generate_debug_images:
            self.draw_detected_crop_rows_on_segmented_image(tile)
        self.measure_vegetation_coverage_in_crop_row(tile)
        if tile.tile_boundry:
            self.add_boundary_and_number_to_tile(tile)

class tile_separator:
    def __init__(self):
        self.tile_size = 3000
        self.run_specific_tile = None
        self.run_specific_tileset = None

        self.resolution = None
        self.crs = None       
        self.left = None
        self.top = None

        # To pass to the crop_row_detector
        self.generate_debug_images = None
        self.tile_boundry = None
        self.expected_crop_row_distance = 20
        self.output_tile_location = None
        self.threshold_level = None
        self.filename_orthomosaic = None

    # Seperating into tiles and running crop rows on tiles
    def main(self, filename_segmented_orthomosaic):
        output_directory = os.path.dirname(self.output_tile_location)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        self.process_orthomosaic(filename_segmented_orthomosaic)

    def process_orthomosaic(self, filename_segmented_orthomosaic):
        start_time = time.time()
        self.calculate_crop_rows_in_orthomosaic(filename_segmented_orthomosaic)
        proc_time = time.time() - start_time
        # print('Calculation of color distances: ', proc_time)

    def calculate_crop_rows_in_orthomosaic(self, filename_segmented_orthomosaic):
        with rasterio.open(filename_segmented_orthomosaic) as src:
            self.resolution = src.res
            self.crs = src.crs
            self.left = src.bounds[0]
            self.top = src.bounds[3]

        if self.filename_orthomosaic is not None:
            with rasterio.open(self.filename_orthomosaic) as src:
                self.resolution = src.res
                self.crs = src.crs
                self.left = src.bounds[0]
                self.top = src.bounds[3]

        processing_tiles = self.get_processing_tiles(filename_segmented_orthomosaic,
                                                     self.tile_size)
        if self.filename_orthomosaic is not None:
            self.tiles_plot = self.get_processing_tiles(self.filename_orthomosaic,
                                                     self.tile_size)

        for tile_number, tile in enumerate(tqdm(processing_tiles)):
            tile.tile_number = tile_number
            if self.run_specific_tileset is not None:
                if tile_number >= self.run_specific_tileset[0] and tile_number <= self.run_specific_tileset[1]:
                    img = read_tile(filename_segmented_orthomosaic, tile)
                    self.process_tile(img, tile_number, tile)
                elif self.run_specific_tile is not None:
                    if tile_number in self.run_specific_tile:
                        img = read_tile(filename_segmented_orthomosaic, tile)
                        self.process_tile(img, tile_number, tile)
            elif self.run_specific_tile is not None:
                if tile_number in self.run_specific_tile:
                    img = read_tile(filename_segmented_orthomosaic, tile)
                    self.process_tile(img, tile_number, tile)
            else:
                img = read_tile(filename_segmented_orthomosaic, tile)
                self.process_tile(img, tile_number, tile)

    def get_processing_tiles(self, filename_segmented_orthomosaic, tile_size):
        """
        Generate a list of tiles to process, including a padding region around
        the actual tile.
        Takes care of edge cases, where the tile does not have adjacent tiles in
        all directions.
        """
        processing_tiles, st_width, st_height = self.define_tiles(
            filename_segmented_orthomosaic, 0.01, tile_size, tile_size)

        no_r = np.max([t.tile_position[0] for t in processing_tiles])
        no_c = np.max([t.tile_position[1] for t in processing_tiles])

        half_overlap_c = (tile_size-st_width)/2
        half_overlap_r = (tile_size-st_height)/2

        for tile in processing_tiles:
            tile.processing_range = [[half_overlap_r, tile_size - half_overlap_r],
                                     [half_overlap_c, tile_size - half_overlap_c]]
            if tile.tile_position[0] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[0] == no_r:
                tile.processing_range[0][1] = tile_size
            if tile.tile_position[1] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[1] == no_c:
                tile.processing_range[0][1] = tile_size

        return processing_tiles
    
    def define_tiles(self, filename_segmented_orthomosaic, overlap, height, width):
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.
        """

        with rasterio.open(filename_segmented_orthomosaic) as src:
            columns = src.width
            rows = src.height

        last_position = (rows - height, columns - width)

        n_height = np.ceil(rows / (height * (1 - overlap))).astype(int)
        n_width = np.ceil(columns / (width * (1 - overlap))).astype(int)

        step_height = np.trunc(last_position[0] / (n_height - 1)).astype(int)
        step_width = np.trunc(last_position[1] / (n_width - 1)).astype(int)

        tiles = []
        for r in range(0, n_height):
            for c in range(0, n_width):
                pos = [r, c]
                if r == (n_height - 1):
                    tile_r = last_position[0]
                else:
                    tile_r = r * step_height
                if c == (n_width - 1):
                    tile_c = last_position[1]
                else:
                    tile_c = c * step_width
                tiles.append(Tile((tile_r, tile_c), pos, height, width))

        return tiles, step_width, step_height

    def process_tile(self, segmented_img, tile_number, tile):
        
        width = tile.size[1]
        height = tile.size[0]
        self.tile_number = tile_number
        # run row detection on tile
        if segmented_img.shape[0] == 1:
            self.gray = segmented_img.reshape(segmented_img.shape[1], segmented_img.shape[2])
            if self.filename_orthomosaic is not None:
                self.img = read_tile(self.filename_orthomosaic, self.tiles_plot[tile_number])
            else: 
                self.img = segmented_img.reshape(segmented_img.shape[1], segmented_img.shape[2])
            self.gray_reduce()
        else:
            self.img = segmented_img
            self.convert_to_grayscale()
        
        self.apply_hough_lines()
        self.determine_dominant_row()
        self.plot_counts_in_hough_accumulator_with_direction()
        self.determine_and_plot_offsets_of_crop_rows_with_direction()
        self.draw_detected_crop_rows_on_input_image()
        if self.generate_debug_images:
            self.draw_detected_crop_rows_on_segmented_image()
        self.measure_vegetation_coverage_in_crop_row(tile)
        if self.tile_boundry:
            self.add_boundary_and_number_to_tile()

        # save results
        tile.ulc_global = [
                self.top - (tile.ulc[0] * self.resolution[0]), 
                self.left + (tile.ulc[1] * self.resolution[1])]
        transform = Affine.translation(
            tile.ulc_global[1] + self.resolution[0] / 2,
            tile.ulc_global[0] - self.resolution[0] / 2) * \
            Affine.scale(self.resolution[0], -self.resolution[0])

        print("tile_position: ", tile.tile_position)
        print("tile.ulc: ", tile.ulc)
        print("tile.ulc_global: ", tile.ulc_global)
        print("tile_number: ", tile.tile_number)

        # optional save of results - just lob detection and thresholding result
        self.save_results(self.img, tile_number,
                          self.resolution, height, width, self.crs, transform)

    def save_results(self, img, tile_number, res, height, width, crs, transform):
        if self.output_tile_location is not None:
            name_mahal_results = f'{ self.output_tile_location }/mahal{ tile_number:04d}.tiff'
            img_to_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            channels = img_to_save.shape[2]
            temp_to_save = img_to_save.transpose(2, 0, 1) 
            new_dataset = rasterio.open(name_mahal_results,
                                        'w',
                                        driver='GTiff',
                                        res=res,
                                        height=height,
                                        width=width,
                                        count=channels,
                                        dtype=temp_to_save.dtype,
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(temp_to_save)
            new_dataset.close()


parser = argparse.ArgumentParser(description = "Detect crop rows in segmented image")
parser.add_argument('segmented_orthomosaic', 
                    help='Path to the segmented_orthomosaic that you want to process.')
parser.add_argument('--orthomosaic',
                    help='Path to the orthomosaic that you want to plot on. '
                    'if not set, the segmented_orthomosaic will be used.')
parser.add_argument('--tile_size',
                    default=3000,
                    type = int,
                    help='The height and width of tiles that are analyzed. '
                         'Default is 3000.')
parser.add_argument('--output_tile_location', 
                    default='output/mahal',
                    help='The location in which to save the mahalanobis tiles.')
parser.add_argument('--generate_debug_images', 
                    default = False,
                    type = bool, 
                    help = "If set to true, debug images will be generated, defualt is False")
parser.add_argument('--tile_boundry',
                    default = False,
                    type = bool,
                    help='if set to true will plot a boundry on each tile ' 
                    'and the tile number on the til, is default False.')
parser.add_argument('--run_specific_tile',
                    nargs='+',
                    type=int,
                    help='If set, only run the specific tile numbers. '
                    '(--run_specific_tile 16 65) will run tile 16 and 65.')
parser.add_argument('--run_specific_tileset',
                    nargs='+',
                    type=int,
                    help='takes to inputs like (--from_specific_tile 16 65). '
                    'this will run every tile from 16 to 65.')
parser.add_argument('--expected_crop_row_distance',
                    default=20,
                    type=int,
                    help='The expected distance between crop rows in pixels, default is 20.')
args = parser.parse_args()

crd = crop_row_detector()
crd.generate_debug_images = args.generate_debug_images
crd.tile_boundry = args.tile_boundry
crd.run_specific_tile = args.run_specific_tile
crd.run_specific_tileset = args.run_specific_tileset
crd.tile_size = args.tile_size
crd.expected_crop_row_distance = args.expected_crop_row_distance
crd.output_tile_location = args.output_tile_location
crd.filename_orthomosaic = args.orthomosaic
crd.threshold_level = 12
crd.main(args.segmented_orthomosaic)



# python3 crop_row_detector.py rødsvingel/rødsvingel.tif --orthomosaic rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundry True --generate_debug_images True --run_specific_tile 16
# gdal_merge.py -o rødsvingel/rødsvingel_crd.tif -a_nodata 255 rødsvingel/tiles_crd/mahal*.tiff


