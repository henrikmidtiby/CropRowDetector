import cv2
import numpy as np
from skimage.transform import hough_line
from skimage import filters
from path import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from icecream import ic
from pybaselines import Baseline, utils
from datetime import datetime

import cython_test

import time


# python3 crop_row_detector.py 2023-05-01_Alm_rajgraes_cleopatra_jsj_imput_image --generate_debug_images True --debug_image_folder output 


class crop_row_detector:
    def __init__(self):
        self.generate_debug_images = None
        self.output_folder = None
        self.img = None
        self.thresholded_image = None
        self.h = None
        self.theta = None
        self.d = None
        self.max_value = None
        self.direction_with_most_energy_idx = None
        self.direction = None
        self.peaks = None
        self.exg = None
        self.gray = None
        self.HSV_gray = None
        self.save_with_name = None
        self.date_time = str(datetime.now().strftime("%Y_%m_%d_%H:%M"))

    def ensure_parent_directory_exist(self, path):
        temp_path = Path(path).parent
        if not temp_path.exists():
            temp_path.mkdir()

    def write_image_to_file(self, output_path, img):
        if self.generate_debug_images:
            if self.save_with_name:
                self.ensure_parent_directory_exist(self.output_folder + "/" + output_path)
                cv2.imwrite(self.output_folder + "/" + output_path, img)
            else:
                self.ensure_parent_directory_exist(self.date_time + "/" + output_path)
                cv2.imwrite(self.date_time + "/" + output_path, img)

    def write_plot_to_file(self, output_path):
        if self.generate_debug_images:
            if self.save_with_name:
                self.ensure_parent_directory_exist(self.output_folder + "/" + output_path)
                plt.savefig(self.output_folder + "/" + output_path, dpi=300)
            else:
                self.ensure_parent_directory_exist(self.date_time + "/" + output_path)
                plt.savefig(self.date_time + "/" + output_path, dpi=300)

    def segment_image(self, img):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]
        exg = 2.0 * g - 1.0 * r - 1.0 * b
        threshold = filters.threshold_otsu(exg)
        ic(threshold)
        #threshold = 7
        thresholded_image = exg > threshold
        self.thresholded_image = thresholded_image
        ic(thresholded_image.shape)

        thres = cv2.minMaxLoc(g)[1]

        test = g.copy()
        for x in range(0, test.shape[0]):
            for y in range(0, test.shape[1]):
                if test[x,y] < thres * 0.2:
                    test[x,y] = test[x,y]
                else:
                    test[x,y] = 0

        self.exg = test #cv2.minMaxLoc(g)[1] - g
    
    def load_and_segment_image(self, filename):
        img = cv2.imread(filename)
        self.segment_image(img)
        self.img = img
        self.write_image_to_file("31_loaded_image.png", self.img)
        self.write_image_to_file("32_segmented_image.png", self.thresholded_image.astype(np.uint8) * 255)
        self.write_image_to_file("32_test.png", self.exg)#.astype(np.uint8))
        self.write_image_to_file("32_segmented_image_inverse.png", 255 - self.thresholded_image.astype(np.uint8) * 255)

    def apply_hough_lines(self):
        # Apply the hough transform
        number_of_angles = 8*360
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
        min_h, max_h, _, _ = cv2.minMaxLoc(self.HSV_gray)
        ic(max_h)
        start_time = time.time()
        h, theta, d = cython_test.hough_line(self.HSV_gray, theta=tested_angles)
        #h, theta, d = hough_line(self.HSV_gray, theta=tested_angles)
        print("--- %s seconds ---" % (time.time() - start_time))
        #h, theta, d = hough_line(self.gray, theta=tested_angles)
        #ic(d.shape)
        #ic(d)
        

        self.theta = theta
        self.d = d
        self.h = h
        self.max_value = cv2.minMaxLoc(h)[1]
        #self.running_average(h)
        self.write_image_to_file("35_hough_image.png", 255 * self.h / self.max_value)

    def determine_dominant_row(self):
        # Determine the dominant row direction
        direction_response = np.sum(np.square(self.h), axis=0)
        baseline_fitter = Baseline(self.theta*180/np.pi, check_finite=False)
        plt.figure(figsize=(16, 9))
        plt.plot(self.theta*180/np.pi, np.log(direction_response), color='blue')
        plt.plot(self.theta*180/np.pi, baseline_fitter.mor(np.log(direction_response), half_window=30)[0], color='orange')
        self.write_plot_to_file("36_direction_energies.png")
        plt.close()
        direction_with_most_energy_idx = np.argmax(direction_response)
        direction = self.theta[direction_with_most_energy_idx]

        # Normalize the direction response
        plt.figure(figsize=(16, 9))
        Direc_energi = np.log(direction_response) - baseline_fitter.mor(np.log(direction_response), half_window=30)[0]
        Max = np.max(Direc_energi)
        ic(Max)
        #plt.plot(self.theta*180/np.pi, np.log(direction_response) - baseline_fitter.mor(np.log(direction_response), half_window=30)[0])
        plt.plot(self.theta*180/np.pi, Direc_energi/Max)
        self.write_plot_to_file("37_direction_energies_normalized.png")
        plt.close()

        self.direction_with_most_energy_idx = direction_with_most_energy_idx
        self.direction = direction

    def plot_counts_in_hough_accumulator_with_direction(self):
        plt.figure(figsize=(16, 9))
        plt.plot(self.h[:, self.direction_with_most_energy_idx])
        self.write_plot_to_file("38_row_offsets.png")
        plt.close()

    def determine_and_plot_offsets_of_crop_rows_with_direction(self):
        signal = self.h[:, self.direction_with_most_energy_idx]

        # In gimp I have measured the crop row distance to be around 20 px.
        # however I get the best results when this value is set to 30.
        expected_crop_row_distance = 30
        peaks, _ = find_peaks(signal, distance=expected_crop_row_distance / 2)
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
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            # print("y0: ", y0)
            # print("y1: ", y1)
            cv2.line(self.img, (0, int(y0)), (self.img.shape[1], int(y1)), (0, 0, 255), 1)
        self.write_image_to_file("40_detected_crop_rows.png", self.img)

    def draw_detected_crop_rows_on_segmented_image(self):
        temp = 255 * self.thresholded_image.astype(np.uint8)
        segmented_annotated = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        # Draw detected crop rows on the segmented image
        origin = np.array((0, segmented_annotated.shape[1]))
        segmented_annotated = 255 - segmented_annotated
        for peak_idx in self.peaks:
            dist = self.d[peak_idx]
            angle = self.direction
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            cv2.line(segmented_annotated, (0, int(y0)), (segmented_annotated.shape[1], int(y1)), (0, 0, 255), 1)
        self.write_image_to_file("45_detected_crop_rows_on_segmented_image.png", segmented_annotated)

    def run_all(self, img):
        self.load_and_segment_image(img)
        self.convert_to_grayscale()
        self.apply_hough_lines()
        self.determine_dominant_row()
        self.plot_counts_in_hough_accumulator_with_direction()
        self.determine_and_plot_offsets_of_crop_rows_with_direction()
        self.draw_detected_crop_rows_on_input_image()
        self.draw_detected_crop_rows_on_segmented_image()

    def convert_to_grayscale(self):
        HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        HSV_gray = HSV[:,:,2].copy()
        #HSV_gray = 255-HSV_gray
        self.write_image_to_file("02_hsv.png", HSV_gray)
        for i in range(0, HSV_gray.shape[0]):
            for j in range(0, HSV_gray.shape[1]):
                if HSV[i,j,0] > 45 and HSV[i,j,0] < 65:
                    HSV_gray[i,j] = 255-HSV_gray[i,j]
                else:
                    HSV_gray[i,j] = 0
        
        self.write_image_to_file("03_hsv.png", HSV_gray)
        self.HSV_gray = HSV_gray

        """HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        HSV_gray = HSV[:,:,2]
        min_h, max_h, _, _ = cv2.minMaxLoc(HSV[:,:,2])
        ic(max_h)
        HSV_gray = (1-HSV[:,:,2]/max_h)*255
        
        for i in range(0, HSV_gray.shape[0]):
            for j in range(0, HSV_gray.shape[1]):
                if HSV_gray[i,j] > 200:
                    HSV_gray[i,j] = 255#HSV_gray[i,j]
                else:
                    HSV_gray[i,j] = 0
        
        self.write_image_to_file("02_hsv.png", HSV_gray)
        self.HSV_gray = HSV_gray
        ic(HSV_gray.shape)"""

"""
def ensure_parent_directory_exist(path):
    temp_path = Path(path).parent
    if not temp_path.exists():
        temp_path.mkdir()

def write_image_to_file(output_path, img):
    if generate_debug_images:
        ensure_parent_directory_exist(output_path)
        cv2.imwrite(output_path, img)

def write_plot_to_file(output_path):
    if generate_debug_images:
        ensure_parent_directory_exist(output_path)
        plt.savefig(output_path, dpi=300)

def segment_image(self, img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    exg = 2.0 * g - 1.0 * r - 1.0 * b
    threshold = filters.threshold_otsu(exg)
    ic(threshold)
    #threshold = 7
    thresholded_image = exg > threshold
    self.thresholded_image = thresholded_image

def calculate_normalization(width, height, theta_vals):
    xlow = 0
    ylow = 0
    xhigh = width
    yhigh = height
    value_matrix = np.vstack((
        xlow * np.sin(theta_vals) + ylow * np.cos(theta_vals),
        xhigh * np.sin(theta_vals) + ylow * np.cos(theta_vals),
        xlow * np.sin(theta_vals) + yhigh * np.cos(theta_vals),
        xhigh * np.sin(theta_vals) + yhigh * np.cos(theta_vals)
    ))
    
    min_val = np.min(value_matrix, axis = 0)
    max_val = np.max(value_matrix, axis = 0)
    width_of_baseline = max_val - min_val
    return 1 / width_of_baseline

def detect_crop_rows(filename):
    img = cv2.imread(filename)
    write_image_to_file("output/31_loaded_image.png", img)

    thresholded_image = segment_image(img)
    write_image_to_file("output/32_segmented_image.png", thresholded_image.astype(np.uint8) * 255)
    write_image_to_file("output/32_segmented_image_inverse.png", 255 - thresholded_image.astype(np.uint8) * 255)

    kernal = np.ones((5,5),np.uint8)
    img_erosion = cv2.erode(thresholded_image.astype(np.uint8) * 255, kernal, iterations=2)
    img_dilation = cv2.dilate(img_erosion, kernal, iterations=2)
    write_image_to_file("output/32_segmented_image_tophat.png", img_dilation)

    temp = 255 * thresholded_image.astype(np.uint8)
    segmented_annotated = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

    # Apply the hough transform
    # 8 * 360
    number_of_angles = 8*360
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
    h, theta, d = hough_line(thresholded_image, theta=tested_angles)
    #h, theta, d = hough_line(img_dilation, theta=tested_angles)
    min_value, max_value, _, _ = cv2.minMaxLoc(h)

    # Average hough lines to remove clutter
    hough_corrected = h.copy()
    h_test = h.copy()
    for i in range(0, h_test.shape[0]-2):
        for j in range(0, h_test.shape[1]):
            try:
                #Could divide by 5, but this is faster and gives similar results, as minMaxLoc is used later
                hough_corrected[i,j] = (h_test[i-2,j]+h_test[i-1,j]+h_test[i,j]+h_test[i+1,j]+h_test[i+2,j])
                #hough_corrected[i,j] = (h_test[i,j]+h_test[i+1,j])
                #hough_corrected[i,j] = (h_test[i,j])
            except Exception as e:
                print(e)

    hough_corrected = np.array(hough_corrected)
    min_h, max_h, _, _ = cv2.minMaxLoc(hough_corrected)

    write_image_to_file("output/33_hough_image.png", 255 * hough_corrected / max_h)

    kernal = np.ones((5,5),np.uint8)
    img_erosion = cv2.erode(255 * h / max_value, kernal, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernal, iterations=1)
    write_image_to_file("output/34_hough_image.png", img_dilation)

    write_image_to_file("output/35_hough_image.png", 255 * h / max_value)


    beam_width = calculate_normalization(img.shape[0], img.shape[1], tested_angles)


    # Determine the dominant row direction
    direction_response = np.sum(np.square(hough_corrected), axis=0)
    ic(direction_response)
    plt.figure(figsize=(16, 9))
    plt.plot(theta*180/np.pi, np.log(direction_response), color='blue')
    baseline_fitter = Baseline(theta*180/np.pi, check_finite=False) # Promissing
    plt.plot(theta*180/np.pi, baseline_fitter.mor(np.log(direction_response), half_window=30)[0], color='orange') # Promissing
    #plt.plot(theta*180/np.pi, baseline_fitter.tophat(np.log(direction_response))[0], color='orange')
    #plt.plot(theta*180/np.pi, 0.75 * np.log(beam_width) + 20.5, color='red')
    write_plot_to_file("output/36_direction_energies.png")
    plt.close()
    direction_with_most_energy_idx = np.argmax(direction_response)
    direction = theta[direction_with_most_energy_idx]

    # Normalize the direction response
    plt.figure(figsize=(16, 9))
    #plt.plot(theta*180/np.pi, np.log(direction_response) - np.log(beam_width))
    plt.plot(theta*180/np.pi, np.log(direction_response) - baseline_fitter.mor(np.log(direction_response), half_window=30)[0]) # Promissing
    #plt.plot(theta*180/np.pi, np.log(direction_response) - baseline_fitter.tophat(np.log(direction_response))[0])
    write_plot_to_file("output/37_direction_energies_normalized.png")
    plt.close()

    # Plot counts in the hough accumulator with this direction
    plt.figure(figsize=(16, 9))
    plt.plot(hough_corrected[:, direction_with_most_energy_idx])
    write_plot_to_file("output/38_row_offsets.png")
    plt.close()

    # Determine offsets of crop rows with this direction
    signal = hough_corrected[:, direction_with_most_energy_idx]

    # In gimp I have measured the crop row distance to be around 20 px.
    # however I get the best results when this value is set to 30.
    expected_crop_row_distance = 30
    peaks, _ = find_peaks(signal, distance=expected_crop_row_distance / 2)
    print(peaks)
    plt.figure(figsize=(16, 9))
    plt.plot(signal)
    plt.plot(peaks, signal[peaks], "x")
    plt.plot(np.zeros_like(signal), "--", color="gray")
    write_plot_to_file("output/39_signal_with_detected_peaks.png")
    plt.close()

    # Draw detected crop rows on the input image
    origin = np.array((0, img.shape[1]))
    for peak_idx in peaks:
        dist = d[peak_idx]
        angle = direction
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        # print("y0: ", y0)
        # print("y1: ", y1)
        cv2.line(img, (0, int(y0)), (img.shape[1], int(y1)), (0, 0, 255), 1)
    write_image_to_file("output/40_detected_crop_rows.png", img)

    # Draw detected crop rows on the segmented image
    origin = np.array((0, segmented_annotated.shape[1]))
    segmented_annotated = 255 - segmented_annotated
    for peak_idx in peaks:
        dist = d[peak_idx]
        angle = direction
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        cv2.line(segmented_annotated, (0, int(y0)), (segmented_annotated.shape[1], int(y1)), (0, 0, 255), 1)
    write_image_to_file("output/45_detected_crop_rows_on_segmented_image.png", segmented_annotated)

"""


parser = argparse.ArgumentParser(description = "Detect crop rows in segmented image")
parser.add_argument('filename', 
                    type=str, 
                    help = "filename of image to process")
parser.add_argument('--generate_debug_images', 
                    default = False,
                    type=bool, 
                    help = "If set to true, debug images will be generated, defualt is False")
parser.add_argument('--save_with_name', 
                    default = False,
                    type=bool, 
                    help = 'If set to true, the debug_image_folder can be used to specify the ' 
                           'folder name, if set to false, the folder will be named with the current time')
parser.add_argument('--debug_image_folder', 
                    default = "output",
                    type=str, 
                    help = "folder where debug images will be stored, defualt is output")
args = parser.parse_args()

crd = crop_row_detector()
crd.generate_debug_images = args.generate_debug_images
crd.output_folder = args.debug_image_folder
crd.save_with_name = args.save_with_name
#crd.load_and_segment_image(args.filename)
#crd.convert_to_grayscale()
crd.run_all(args.filename)

#detect_crop_rows(args.filename)





