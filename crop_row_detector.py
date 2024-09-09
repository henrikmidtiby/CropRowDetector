import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import concurrent.futures

#import hough_transform_grayscale # This is a custom implementation of the hough transform
from skimage.transform import hough_line
from scipy.signal import find_peaks
from pybaselines import Baseline
from pathlib import Path
from icecream import ic


import statistics
import traceback

def rSVD(X, r, q, p):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    #cutoff = 2.858 * statistics.median((255 - X).flatten())
    #r = np.max(np.where(S > cutoff))
    #print("cutoff: ", cutoff)
    #print("r: ", r)
    
    
    # step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny, r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)
    
    Q, _ = np.linalg.qr(Z, mode='reduced')

    # step 2: Compute the reduced SVD of Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=False)
    U = Q @ UY
    return U, S, VT, r

def reconstruct(U, S, VT, r):
    return U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:]



class crop_row_detector:
    def __init__(self):
        self.generate_debug_images = False
        self.tile_boundry = False
        self.threshold_level = 10
        self.expected_crop_row_distance = 20
        # This class is just a crop row detctor in form of a collection of functions, 
        # all of the information is stored in the information class Tile. 


    def ensure_parent_directory_exist(self, path):
        temp_path = Path(path).parent
        if not temp_path.exists():
            temp_path.mkdir(parents = True)

    def get_debug_output_filepath(self, output_path, tile):
        return tile.output_tile_location + "/debug_images/" + f'{tile.tile_number}' + "/" + output_path

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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  
                                        filterSize) 
        h = cv2.morphologyEx(h, cv2.MORPH_TOPHAT, kernel)
        return h
    
    def blur_image(self, h):
        # Blur image using a 5 x 1 average filter
        kernel = np.ones((5,1), np.float32) / 5
        h = cv2.filter2D(h, -1, kernel)
        return h

    def apply_hough_lines(self, tile):
        number_of_angles = 8*360
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
        
        #scipy's implementation er anvendt, da denne for nu er hurtigere
        #self.h, self.theta, self.d = hough_transform_grayscale.hough_line(self.gray, theta=tested_angles)
        h, tile.theta, tile.d = hough_line(tile.gray, theta=tested_angles)
        h = h.astype(np.float32)

        h = self.normalize_array(h)
        self.write_image_to_file("33_hough_image.png", 255 * h, tile)

        
        h = self.blur_image(h)
        h = self.normalize_array(h)
        self.write_image_to_file("34_hough_image_blurred.png", 255 * h, tile)

        h = self.apply_top_hat(h, tile)
        
        tile.h = self.normalize_array(h)

        self.write_image_to_file("35_hough_image_tophat.png", 255 * h, tile)

    
    def normalize_array(self, arr):
        max = cv2.minMaxLoc(arr)[1]
        if max > 0:
            arr = arr/max
        else:
            # This is implemented to stop the padding tiles from being 0 
            # and therefore throwing an error when using np.log, as log(0) is undefined.
            arr = arr + 10e-10
        return arr

    def determine_dominant_direction(self, tile):
        baseline_fitter = Baseline(tile.theta*180/np.pi, check_finite=False)

        # There are 4 different ways to determine the dominant row, as seen below.
        tile.direction_response = np.sum(np.square(tile.h), axis=0)
        tile.log_direc = np.log(tile.direction_response)
        tile.log_direc_baseline = np.log(tile.direction_response) - baseline_fitter.mor(np.log(tile.direction_response), half_window=30)[0]
        tile.direc_baseline = tile.direction_response - baseline_fitter.mor(tile.direction_response, half_window=30)[0]
        

        tile.direction_with_most_energy_idx = np.argmax(tile.direc_baseline)
        tile.direction = tile.theta[tile.direction_with_most_energy_idx]
        
        
        self.plot_direction_energies(tile)
        

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
        

    def plot_direction_energies(self, tile):
        plt.figure(figsize=(16, 9))
        self.plot_direction_response_and_maximum(tile, tile.log_direc, 
                                                 'blue', 'log of direction response')
        self.plot_direction_response_and_maximum(tile, tile.direc_baseline, 
                                                 'green', 'direction response - baseline')
        self.plot_direction_response_and_maximum(tile, tile.log_direc_baseline, 
                                                 'orange', 'log of direction response - baseline')
        self.plot_direction_response_and_maximum(tile, tile.direction_response, 
                                                 'red', 'direction response')
        plt.legend()
        self.write_plot_to_file("36_direction_energies.png", tile)
        plt.close()

    def plot_direction_response_and_maximum(self, tile, direction_response, color, label):
        plt.plot(tile.theta*180/np.pi, direction_response, 
                 color=color, label=label)
        plt.axvline(x=tile.theta[np.argmax(direction_response)]*180/np.pi, 
                    color=color, linestyle='dashed')

    
    def determine_offsets_of_crop_rows(self, tile):
        tile.signal = tile.h[:, tile.direction_with_most_energy_idx]
        tile.peaks, _ = find_peaks(tile.signal, 
                                   distance=tile.expected_crop_row_distance / 2, 
                                   prominence=0.01)
        self.plot_row_offset(tile)
        self.plot_row_offset_with_peaks(tile)
    
    
    def plot_row_offset(self, tile):
        plt.figure(figsize=(16, 9))
        plt.plot(tile.signal, color='blue')
        self.write_plot_to_file("38_row_offsets.png", tile)
        plt.close()


    def plot_row_offset_with_peaks(self, tile):
        plt.figure(figsize=(16, 9))
        plt.plot(tile.signal)
        plt.plot(tile.peaks, tile.signal[tile.peaks], "x")
        plt.plot(np.zeros_like(tile.signal), "--", color="gray")
        self.write_plot_to_file("39_row_offsets_with_detected_peaks.png", tile)
        plt.close()

    def determine_line_ends_of_crop_rows(self, tile):
        tile.vegetation_lines = []
        prev_peak_dist = 0

        for peak_idx in tile.peaks:
            dist = tile.d[peak_idx]
            
            angle = tile.direction

            self.fill_in_gaps_in_detected_crop_rows(dist, prev_peak_dist, angle, tile)
            line_ends = self.get_line_ends_within_image(dist, angle, tile.img_constant)

            prev_peak_dist = dist
            tile.vegetation_lines.append(line_ends)

        vegetation_lines = []
        for line_ends in tile.vegetation_lines:
            if len(line_ends) == 2:
                vegetation_lines.append(line_ends)
        tile.vegetation_lines = vegetation_lines


    def draw_detected_crop_rows_on_input_image_and_segmented_image(self, tile):
        for line_ends in tile.vegetation_lines:
            try:
                self.draw_crop_row(tile.img, line_ends)
                self.draw_crop_row(tile.gray_inverse, line_ends)
            except Exception as e:
                print(e)
                ic(line_ends)

        self.write_image_to_file("40_detected_crop_rows.png", tile.img, tile)
        self.write_image_to_file("45_detected_crop_rows_on_segmented_image.png", tile.gray_inverse, tile)

    def draw_crop_row(self, image, line_ends):
        cv2.line(image, (line_ends[0][0], line_ends[0][1]), 
                 (line_ends[1][0], line_ends[1][1]), 
                 (0, 0, 255), 1)

    def add_boundary_and_number_to_tile(self, tile):
        if tile.tile_boundry:
            cv2.line(tile.img, (0, 0), (tile.img.shape[1]-1, 0), (0, 0, 255), 1)
            cv2.line(tile.img, (0, tile.img.shape[0]-1), (tile.img.shape[1]-1, tile.img.shape[0]-1), (0, 0, 255), 1)
            cv2.line(tile.img, (0, 0), (0, tile.img.shape[0]-1), (0, 0, 255), 1)
            cv2.line(tile.img, (tile.img.shape[1]-1, 0), (tile.img.shape[1]-1, tile.img.shape[0]-1), (0, 0, 255), 1)
            cv2.putText(tile.img, f'{tile.tile_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def fill_in_gaps_in_detected_crop_rows(self, dist, prev_peak_dist, angle, tile):
        if prev_peak_dist != 0:
            while self.distance_between_two_peaks_is_larger_than_expected(dist, prev_peak_dist, tile):
                prev_peak_dist += tile.expected_crop_row_distance
                line_ends = self.get_line_ends_within_image(prev_peak_dist, angle, tile.img_constant)
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
        
        if line_ends[0][0] > line_ends[1][0]:
            line_ends = [line_ends[1], line_ends[0]]
        
        return line_ends     

    def measure_vegetation_coverage_in_crop_row(self, tile):
        # 1. Blur image with a uniform kernel
        # Approx distance between crop rows is 16 pixels.
        # I would prefer to have a kernel size that is not divisible by two.
        vegetation_map = cv2.blur(tile.gray.astype(np.uint8), (10, 10))
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
                self.find_vegetation_in_crop_row(tile, vegetation_map, missing_plants_image, df_missing_vegetation_list, counter, crop_row)
            except Exception as e:
                print(e)
                traceback.print_exc()
                print("measure_vegetation_coverage_in_crop_row: ")
                print("tile: ", tile.tile_number)


             
        if df_missing_vegetation_list:
            DF_combined = pd.concat(df_missing_vegetation_list)
        
        if tile.generate_debug_images:
            filename = self.get_debug_output_filepath("68_vegetation_samples.csv", tile)       
            DF_combined.to_csv(filename)
        self.write_image_to_file("67_missing_plants_in_crop_line.png", missing_plants_image, tile)

        # 3. Export to a csv file, include the following information
        #    - row number and offset
        #    - pixel coordinates
        #    - vegetation coverage  


    def find_vegetation_in_crop_row(self, tile, vegetation_map, missing_plants_image, df_missing_vegetation_list, counter, crop_row):
        x_sample_coords, y_sample_coords = self.calculate_x_and_y_sample_cords_along_crop_row(tile, crop_row)
                
        DF = self.create_data_structure_containing_crop_row(tile, vegetation_map, counter, x_sample_coords, y_sample_coords)
        df_missing_vegetation_list.append(DF)
                
        self.plot_points_without_vegetation_on_crop_row(tile, missing_plants_image, DF)

    def plot_points_without_vegetation_on_crop_row(self, tile, missing_plants_image, DF):
        threshold_vegetation = 60
        missing_plants = DF[DF['vegetation'] < threshold_vegetation]
        for index, location in missing_plants.iterrows():
            cv2.circle(missing_plants_image, 
                            (int(location['x'] - tile.size[0]*tile.tile_position[1]), 
                             int(location['y'] - tile.size[1]*tile.tile_position[0])), 
                            2, (255, 255, 0), -1)

    def create_data_structure_containing_crop_row(self, tile, vegetation_map, counter, x_sample_coords, y_sample_coords):
        vegetation_samples = cv2.remap(vegetation_map, 
                                            x_sample_coords.astype(np.float32), 
                                            y_sample_coords.astype(np.float32), 
                                            cv2.INTER_LINEAR)

        DF = pd.DataFrame({'tile': tile.tile_number,
                                'row': counter, 
                                'x': x_sample_coords + tile.size[0]*tile.tile_position[1], 
                                'y': y_sample_coords + tile.size[1]*tile.tile_position[0], 
                                'vegetation': vegetation_samples.transpose()[0]})
                        
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
    

    def convert_segmented_image_to_binary(self, tile):
        binary = tile.segmented_img.copy()
        for i in range(0, binary.shape[0]):
            for j in range(0, binary.shape[1]):
                if binary[i,j] < tile.threshold_level:
                    binary[i,j] = 255
                else:
                    binary[i,j] = 0
        tile.gray = binary
        tile.gray_inverse = 255 - binary

    def load_tile_with_data_needed_for_crop_row_detection(self, tile):
        tile.generate_debug_images = self.generate_debug_images
        tile.tile_boundry = self.tile_boundry
        tile.threshold_level = self.threshold_level
        # In gimp I have measured the crop row distance to be around 20 px.
        # however I get the best results when this value is set to 30.
        tile.expected_crop_row_distance = self.expected_crop_row_distance

        tile.gray = None
        tile.gray_inverse = None

        # Data for the crop row detecter
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
        assert tile_segmented.img.shape[0] == 1, "The segmented image has more then one color chanal."
        assert tile_plot.ulc == tile_segmented.ulc, "The two tiles are not the same location."
        
        tile_segmented.segmented_img = tile_segmented.img.reshape(tile_segmented.img.shape[1], tile_segmented.img.shape[2]).copy()
        
        if tile_plot.img.shape[0] == 1:
            tile_plot.img = tile_plot.img.reshape(tile_plot.img.shape[1], tile_plot.img.shape[2]).copy()
        tile_segmented.img = tile_plot.img.copy()
        tile_segmented.img_constant = tile_plot.img.copy()


    def main(self, tiles_segmented, tiles_plot):

        for list_iter, tile in enumerate(tiles_segmented):
            self.load_tile_with_data_needed_for_crop_row_detection(tile)
            self.combine_segmented_and_original_tile(tile, tiles_plot[list_iter])

        """for tile in tiles_segmented:
            r = 10
            q = 1
            p = 1
            self.write_image_to_file("98_detected_crop_rows_on_segmented_image.png", tile.segmented_img, tile)
            U, S, VT, r = rSVD(tile.segmented_img, r, q, p)
            tile.segmented_img = reconstruct(U, S, VT, r)
            self.write_image_to_file("99_detected_crop_rows_on_segmented_image.png", tile.segmented_img, tile)"""
        
        """for tile in tiles_segmented:
            tile.save_segmented = tile.segmented_img.copy()
            for i in range(0, 20):
                
                r = i*2
                q = 1
                p = 1
                U, S, VT = rSVD(tile.segmented_img, r, q, p)
                tile.segmented_img = reconstruct(U, S, VT, r)
                self.convert_segmented_image_to_binary(tile)
                self.write_image_to_file(str(i) +"_detected_crop_rows_on_segmented_image.png", 255 - tile.gray, tile)
                tile.segmented_img = tile.save_segmented.copy()"""
        

        start = time.time()
        total_results = []
        if len(tiles_segmented) > 20:
            for i in range(0, int(len(tiles_segmented)/20)+1):
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    result = executor.map(self.detect_crop_rows, tiles_segmented[i*20:i*20+20])
                for res in result:
                    total_results.append(res)
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                total_results = executor.map(self.detect_crop_rows, tiles_segmented)
        print("Time to run all tiles: ", time.time() - start)
        tiles_segmented = list(total_results)

        self.create_csv_of_row_information(tiles_segmented)

    def detect_crop_rows(self, tile):
        t1 = time.time()
        try:
            self.convert_segmented_image_to_binary(tile)
            self.apply_hough_lines(tile)
            self.determine_dominant_direction(tile)
            self.determine_offsets_of_crop_rows(tile)
            self.determine_line_ends_of_crop_rows(tile)
            self.draw_detected_crop_rows_on_input_image_and_segmented_image(tile)
            self.measure_vegetation_coverage_in_crop_row(tile)
            self.add_boundary_and_number_to_tile(tile)
            tile.save_tile()
            print(f"Time to run tile: ", time.time() - t1)
        except Exception as e:
            ic(e)
        return tile

    def create_csv_of_row_information(self, tiles_segmented):
        row_information = []
        
        for tile in tiles_segmented:
            if tile.direction < 0:
                tile.direction = np.pi + tile.direction
            for row_number, row in enumerate(tile.vegetation_lines):
                row_information.append([tile.tile_number, 
                                       tile.tile_position[0],
                                       tile.tile_position[1],
                                       tile.direction,
                                       row_number,  
                                       row[0][0], 
                                       row[0][1],
                                       row[1][0], 
                                       row[1][1]])

        DF_row_information = pd.DataFrame(row_information, columns=['tile', 'x_position', 'y_position', 'angle', 'row', 'x_start', 'y_start', 'x_end', 'y_end'])

        csv_path = tiles_segmented[0].output_tile_location
        DF_row_information.to_csv(csv_path + "/row_information.csv")

