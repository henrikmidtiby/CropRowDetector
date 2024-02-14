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

import time


# python3 crop_row_detector.py 2023-05-01_Alm_rajgraes_cleopatra_jsj_imput_image --generate_debug_images True --debug_image_folder output 

class Tile:
    def __init__(self, start_point, position, height, width):
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.ulc_global = []
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]


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
    def __init__(self):
        self.tile_number = None
        self.generate_debug_images = None
        self.tile_boundry = None
        self.img = None
        self.h = None
        self.theta = None
        self.d = None
        self.direction_with_most_energy_idx = None
        self.direction = None
        self.peaks = None
        self.gray = None
        self.threshold_level = 10
        # In gimp I have measured the crop row distance to be around 20 px.
        # however I get the best results when this value is set to 30.
        self.expected_crop_row_distance = 20 # 30


        # opening and processing the image
        self.filename_orthomosaic = None
        self.output_tile_location = None
        self.resolution = None
        self.crs = None       
        self.left = None
        self.top = None
        self.tile_size = 3000
        self.tiles_plot = None
        self.run_specific_tile = None


    def ensure_parent_directory_exist(self, path):
        temp_path = Path(path).parent
        if not temp_path.exists():
            temp_path.mkdir()

    def write_image_to_file(self, output_path, img): 
        if self.generate_debug_images:
            path = self.get_debug_output_filepath(output_path)
            self.ensure_parent_directory_exist(path)
            cv2.imwrite(path, img)

    def get_debug_output_filepath(self, output_path):
        return self.output_tile_location + "/debug_images/" + f'{self.tile_number}' + "/" + output_path

    def write_plot_to_file(self, output_path):
        if self.generate_debug_images:
            path = self.get_debug_output_filepath(output_path)
            self.ensure_parent_directory_exist(path)
            plt.savefig(path, dpi=300)
    
    def apply_hough_lines(self):
        # Apply the hough transform
        number_of_angles = 8*360
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
        self.h, self.theta, self.d = hough_transform_grayscale.hough_line(self.gray, theta=tested_angles)

        filterSize = (self.expected_crop_row_distance, self.expected_crop_row_distance) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  
                                        filterSize) 
        
        h_temp = self.h.copy()
        
        for i in range(0, self.h.shape[1]):
            h_split = self.h[:,i].copy()
            # Applying the Top-Hat operation 
            tophat_img = cv2.morphologyEx(h_split,  
                                    cv2.MORPH_TOPHAT, 
                                    kernel) 
            for j in range(0, self.h.shape[0]):
                h_temp[j,i] = tophat_img[j]
        
        self.h = h_temp
        
        temp = cv2.minMaxLoc(self.h)[1]
        if temp > 0:
            self.h = self.h/temp
        else:
            self.h = self.h + 0.01

        self.write_image_to_file("35_hough_image.png", 255 * self.h)

        # Blur image using a 5 x 1 average filter
        #kernel = np.ones((5,1), np.float32) / 5
        #self.h = cv2.filter2D(self.h, -1, kernel)
        self.write_image_to_file("35_hough_image_blurred.png", 255 * self.h)


    def determine_dominant_row(self):
        # Determine the dominant row direction
        direction_response = np.sum(np.square(self.h), axis=0)
        baseline_fitter = Baseline(self.theta*180/np.pi, check_finite=False)
        
        # Normalize the direction response
        Direc_energi = np.log(direction_response) - baseline_fitter.mor(np.log(direction_response), half_window=30)[0]
        max = np.max(Direc_energi)
        self.direction_with_most_energy_idx = np.argmax(direction_response)#Direc_energi)
        self.direction = self.theta[self.direction_with_most_energy_idx]
        
        # Plot the direction response and normalized direction response
        plt.figure(figsize=(16, 9))
        plt.plot(self.theta*180/np.pi, np.log(direction_response), color='blue')
        self.write_plot_to_file("36_direction_energies.png")
        plt.plot(self.theta*180/np.pi, Direc_energi, color='orange')
        self.write_plot_to_file("36_direction_energies_2.png")
        plt.close()

        plt.figure(figsize=(16, 9))
        plt.plot(self.theta*180/np.pi, 
                 Direc_energi, 
                 color='orange')
        if max != 0:
            plt.plot(self.theta*180/np.pi, 
                     Direc_energi/max,
                     color='blue')
        self.write_plot_to_file("37_direction_energies_normalized.png")
        plt.close()

    def plot_counts_in_hough_accumulator_with_direction(self):
        plt.figure(figsize=(16, 9))
        plt.plot(self.h[:, self.direction_with_most_energy_idx])
        self.write_plot_to_file("38_row_offsets.png")
        plt.close()

    def determine_and_plot_offsets_of_crop_rows_with_direction(self):
        signal = self.h[:, self.direction_with_most_energy_idx]

        
        peaks, _ = find_peaks(signal, distance=self.expected_crop_row_distance / 2)
        self.peaks = peaks
        plt.figure(figsize=(16, 9))
        plt.plot(signal)
        plt.plot(peaks, signal[peaks], "x")
        plt.plot(np.zeros_like(signal), "--", color="gray")
        self.write_plot_to_file("39_signal_with_detected_peaks.png")
        plt.close()

    def draw_detected_crop_rows_on_input_image(self):
        # Draw detected crop rows on the input image
        origin = np.array((0, self.img.shape[1])) 
        for peak_idx in self.peaks:
            dist = self.d[peak_idx]
            angle = self.direction
            temp = self.get_line_ends_within_image(dist, angle, self.img)
            # print("temp: ", temp)
            try:
                cv2.line(self.img, (temp[0][0], temp[0][1]), 
                         (temp[1][0], temp[1][1]), (0, 0, 255), 1)
            except Exception as e:
                print(e)
                ic(temp)

        if self.tile_boundry:
            self.add_boundary_and_number_to_tile()
        self.write_image_to_file("40_detected_crop_rows.png", self.img)

    def add_boundary_and_number_to_tile(self):
        cv2.line(self.img, (0, 0), (self.img.shape[1]-1, 0), (0, 0, 255), 1)
        cv2.line(self.img, (0, self.img.shape[0]-1), (self.img.shape[1]-1, self.img.shape[0]-1), (0, 0, 255), 1)
        cv2.line(self.img, (0, 0), (0, self.img.shape[0]-1), (0, 0, 255), 1)
        cv2.line(self.img, (self.img.shape[1]-1, 0), (self.img.shape[1]-1, self.img.shape[0]-1), (0, 0, 255), 1)
        cv2.putText(self.img, f'{self.tile_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def get_line_ends_within_image(self, dist, angle, img):
        x_val_range = np.array((0, img.shape[1]))
        y_val_range = np.array((0, img.shape[0]))
        # x * cos(t) + y * sin(t) = r
        y0, y1 = (dist - x_val_range * np.cos(angle)) / np.sin(angle)
        x0, x1 = (dist - y_val_range * np.sin(angle)) / np.cos(angle)
        temp = []
        if int(y0) > 0 and int(y0) < img.shape[0]:
            temp.append([0, int(y0)])
        if int(y1) > 0 and int(y1) < img.shape[0]:
            temp.append([img.shape[0], int(y1)])
        if int(x0) > 0 and int(x0) < img.shape[1]:
            temp.append([int(x0), 0])
        if int(x1) > 0 and int(x1) < img.shape[1]:
            temp.append([int(x1), img.shape[0]])
        return temp

    def measure_vegetation_coverage_in_crop_row(self):
        # 1. Blur image with a uniform kernel
        # Approx distance between crop rows is 16 pixels.
        # I would prefer to have a kernel size that is not divisible by two.
        temp = self.gray.astype(np.uint8)
        vegetation_map = cv2.blur(temp, (10, 10))
        self.write_image_to_file("60_vegetation_map.png", vegetation_map)

        # 2. Sample pixel values along each crop row
        #    - cv2.remap
        # Hardcoded from earlier run of the algorithm.
        # missing_plants_image = cv2.cvtColor(vegetation_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        missing_plants_image = self.img
        df_missing_vegetation_list = []

        for counter, peak_idx in enumerate(self.peaks):
            try:
                dist = self.d[peak_idx]
                angle = self.direction

                # Determine sample locations
                temp = self.get_line_ends_within_image(dist, angle, self.img)
                start_point = (temp[0][0], temp[0][1])
                distance_between_samples = 1
                end_point = (temp[1][0], temp[1][1])
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

                DF = pd.DataFrame({'idx': peak_idx, 
                                'x': x_sample_coords, 
                                'y': y_sample_coords, 
                                'vegetation': vegetation_samples.transpose()[0]})
                df_missing_vegetation_list.append(DF)

                missing_plants = DF[DF['vegetation'] < 60]
                for index, location in missing_plants.iterrows():
                    cv2.circle(missing_plants_image, 
                            (int(location['x']), int(location['y'])), 
                            2, 
                            (255, 255, 0), 
                            -1)
            except Exception as e:
                print(e)
                
        #filename = self.date_time + "/" + "64_vegetation_samples.csv"
        #DF_combined = pd.concat(df_missing_vegetation_list)
        #F_combined.to_csv(filename)
        self.write_image_to_file("67_missing_plants_in_crop_line.png", missing_plants_image)
    
        # 3. Export to a csv file, include the following information
        #    - row number and offset
        #    - pixel coordinates
        #    - vegetation coverage
        

    def draw_detected_crop_rows_on_segmented_image(self):
        segmented_annotated = self.gray.copy()
        # Draw detected crop rows on the segmented image
        origin = np.array((0, segmented_annotated.shape[1]))
        segmented_annotated = 255 - segmented_annotated
        for peak_idx in self.peaks:
            dist = self.d[peak_idx]
            angle = self.direction
            temp = self.get_line_ends_within_image(dist, angle, self.img)
            try:
                cv2.line(segmented_annotated, 
                            (temp[0][0], temp[0][1]), 
                            (temp[1][0], temp[1][1]), 
                            (0, 0, 255), 1)
            except Exception as e:
                print(e)
                ic(temp)
        self.segmented_annotated = segmented_annotated
        self.write_image_to_file("45_detected_crop_rows_on_segmented_image.png", segmented_annotated)

    def convert_to_grayscale(self):
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
        self.gray = HSV_gray

    def gray_reduce(self):
        gray_temp = self.gray.copy()
        #print(gray_temp.shape)
        for i in range(0, self.gray.shape[0]):
            for j in range(0, self.gray.shape[1]):
                if self.gray[i,j] < self.threshold_level:
                    # gray_temp[i,j] = 255-gray_temp[i,j]
                    gray_temp[i,j] = 255
                else:
                    gray_temp[i,j] = 0
        self.gray = gray_temp

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
            if self.run_specific_tile is not None:
                if tile_number == self.run_specific_tile:
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
        self.measure_vegetation_coverage_in_crop_row()
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
                    default=None,
                    type=int,
                    help='If set, only run the specific tile number.')
args = parser.parse_args()

crd = crop_row_detector()
crd.generate_debug_images = args.generate_debug_images
crd.tile_boundry = args.tile_boundry
crd.run_specific_tile = args.run_specific_tile
crd.tile_size = args.tile_size
crd.output_tile_location = args.output_tile_location
crd.filename_orthomosaic = args.orthomosaic
crd.threshold_level = 12
crd.main(args.segmented_orthomosaic)



# python3 crop_row_detector.py rødsvingel/rødsvingel.tif --orthomosaic data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif --output_tile_location rødsvingel/tiles_crd --tile_size 500 --tile_boundry True --generate_debug_images True
# gdal_merge.py -o rødsvingel/rødsvingel_crd.tif -a_nodata 255 rødsvingel/tiles_crd/mahal*.tiff


