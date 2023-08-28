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
    threshold = filters.threshold_otsu(img[:, :, 1])
    thresholded_image = img[:, :, 1] < threshold
    return thresholded_image


def detect_crop_rows(filename):
    img = cv2.imread(filename)
    write_image_to_file("output/31_loaded_image.png", img)
    ic(img)

    thresholded_image = segment_image(img)
    ic(thresholded_image)
    write_image_to_file("output/32_segmented_image.png", thresholded_image.astype(np.uint8) * 255)

    # Apply the hough transform
    number_of_angles = 8*360
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, number_of_angles)
    h, theta, d = hough_line(thresholded_image, theta=tested_angles)
    min_value, max_value, _, _ = cv2.minMaxLoc(h)
    write_image_to_file("output/35_hough_image.png", 255 * h / max_value)


    # Determine the dominant row direction
    direction_response = np.sum(np.square(h), axis=0)
    plt.plot(theta*180/np.pi, direction_response)
    write_plot_to_file("output/36_direction_energies.png")
    plt.close()
    direction_with_most_energy_idx = np.argmax(direction_response)
    direction = theta[direction_with_most_energy_idx]

    # Plot counts in the hough accumulator with this direction
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
        print("y0: ", y0)
        print("y1: ", y1)
        cv2.line(img, (0, int(y0)), (img.shape[1], int(y1)), (0, 0, 255), 1)
    write_image_to_file("output/40_detected_crop_rows.png", img)



parser = argparse.ArgumentParser(description = "Detect crop rows in segmented image")
parser.add_argument('filename', type=str, 
        help = "filename of image to process")
args = parser.parse_args()

detect_crop_rows(args.filename)
