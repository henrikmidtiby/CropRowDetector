import cv2
import numpy as np
from skimage.transform import hough_line
from skimage import filters
from path import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from icecream import ic

generate_debug_images = True

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


def segment_image(img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    exg = 2.0 * g - 1.0 * r - 1.0 * b
    threshold = filters.threshold_otsu(exg)
    ic(threshold)
    threshold = 7
    thresholded_image = exg > threshold
    return thresholded_image


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

    temp = 255 * thresholded_image.astype(np.uint8)
    segmented_annotated = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

    # Apply the hough transform
    # 8 * 360
    number_of_angles = 8*360
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
    h, theta, d = hough_line(thresholded_image, theta=tested_angles)
    min_value, max_value, _, _ = cv2.minMaxLoc(h)
    write_image_to_file("output/35_hough_image.png", 255 * h / max_value)

    beam_width = calculate_normalization(img.shape[0], img.shape[1], tested_angles)


    # Determine the dominant row direction
    direction_response = np.sum(np.square(h), axis=0)
    plt.figure(figsize=(16, 9))
    plt.plot(theta*180/np.pi, np.log(direction_response))
    plt.plot(theta*180/np.pi, 0.75 * np.log(beam_width) + 20.5, color='red')
    write_plot_to_file("output/36_direction_energies.png")
    plt.close()
    direction_with_most_energy_idx = np.argmax(direction_response)
    direction = theta[direction_with_most_energy_idx]

    # Normalize the direction response
    plt.figure(figsize=(16, 9))
    plt.plot(theta*180/np.pi, np.log(direction_response) - np.log(beam_width))
    write_plot_to_file("output/37_direction_energies_normalized.png")
    plt.close()

    # Plot counts in the hough accumulator with this direction
    plt.figure(figsize=(16, 9))
    plt.plot(h[:, direction_with_most_energy_idx])
    write_plot_to_file("output/38_row_offsets.png")
    plt.close()

    # Determine offsets of crop rows with this direction
    signal = h[:, direction_with_most_energy_idx]

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



parser = argparse.ArgumentParser(description = "Detect crop rows in segmented image")
parser.add_argument('filename', type=str, 
        help = "filename of image to process")
args = parser.parse_args()

detect_crop_rows(args.filename)
