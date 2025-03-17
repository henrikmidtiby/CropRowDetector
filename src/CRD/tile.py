from __future__ import annotations

import cv2
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window


def rasterio_opencv2(image):
    if image.shape[0] >= 3:  # might include alpha channel
        false_color_img = image.transpose(1, 2, 0)
        separate_colors = cv2.split(false_color_img)
        return cv2.merge([separate_colors[2], separate_colors[1], separate_colors[0]])
    else:
        return image


class Tile:
    def __init__(self, start_point, position, height, width, resolution, crs, left, top, orthomosaic_path):
        # Data for the tile
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]

        self.resolution = (resolution[1], resolution[0])
        self.crs = crs
        self.left = left
        self.top = top

        self.ulc_global = [
            self.top - (self.ulc[0] * self.resolution[0]),
            self.left + (self.ulc[1] * self.resolution[1]),
        ]
        self.transform = Affine.translation(
            self.ulc_global[1] + self.resolution[1] / 2, self.ulc_global[0] - self.resolution[0] / 2
        ) * Affine.scale(self.resolution[1], -self.resolution[0])

        self.tile_number = None
        self.output_tile_location = None
        self.path_to_orthomosaic = orthomosaic_path

    def read_img(self):
        with rasterio.open(self.path_to_orthomosaic) as src:
            window = Window.from_slices((self.ulc[0], self.lrc[0]), (self.ulc[1], self.lrc[1]))
            im = src.read(window=window)
            im = rasterio_opencv2(im)
        return im

    def save_tile(self, img):
        if self.output_tile_location is not None:
            name_mahal_results = f"{ self.output_tile_location }/mahal{ self.tile_number:04d}.tiff"
            img_to_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            channels = img_to_save.shape[2]
            temp_to_save = img_to_save.transpose(2, 0, 1)
            new_dataset = rasterio.open(
                name_mahal_results,
                "w",
                driver="GTiff",
                res=self.resolution,
                height=self.size[0],
                width=self.size[1],
                count=channels,
                dtype=temp_to_save.dtype,
                crs=self.crs,
                transform=self.transform,
            )
            new_dataset.write(temp_to_save)
            new_dataset.close()
