
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import os
import time

from tile import Tile


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
        self.divide_orthomosaic_into_tiles(filename_segmented_orthomosaic)
        proc_time = time.time() - start_time
        # print('Calculation of color distances: ', proc_time)

    def divide_orthomosaic_into_tiles(self, filename_segmented_orthomosaic):
        with rasterio.open(filename_segmented_orthomosaic) as src:
            self.resolution = src.res
            self.crs = src.crs
            self.left = src.bounds[0]
            self.top = src.bounds[3]


        processing_tiles = self.get_processing_tiles(filename_segmented_orthomosaic,
                                                     self.tile_size)
        if self.filename_orthomosaic is not None:
            tiles_plot = self.get_processing_tiles(self.filename_orthomosaic,
                                                     self.tile_size)

        specified_processing_tiles = self.get_list_of_specified_tiles(processing_tiles)
        
        
        for tile in specified_processing_tiles:
            

            # Initilize the tile with the necessary information
            tile.load_tile_info(self.output_tile_location, 
                                self.generate_debug_images, 
                                self.tile_boundry, 
                                self.threshold_level, 
                                self.expected_crop_row_distance)
            
            
            # Run the initialized crop row detector on the tile
            if self.filename_orthomosaic is not None:
                original_orthomosaic = read_tile(self.filename_orthomosaic, tiles_plot[tile.tile_number])
            else:
                original_orthomosaic = None
            segmented_img = read_tile(filename_segmented_orthomosaic, tile)
            tile.original_orthomosaic = original_orthomosaic
            tile.segmented_img = segmented_img

            #self.process_tile(segmented_img, original_orthomosaic, crd, tile)
        self.specified_tiles = specified_processing_tiles

    # with the given command line arguments, this function will return a list of tiles that should be processed
    def get_list_of_specified_tiles(self, tile_list):
        specified_tiles = []
        for tile_number, tile in enumerate(tile_list):
            if self.run_specific_tileset is not None or self.run_specific_tile is not None:
                if self.run_specific_tileset is not None:
                    if tile_number >= self.run_specific_tileset[0] and tile_number <= self.run_specific_tileset[1]:
                        specified_tiles.append(tile)
                if self.run_specific_tile is not None:
                    if tile_number in self.run_specific_tile:
                        specified_tiles.append(tile)
            else:
                specified_tiles.append(tile)

        return specified_tiles


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

        for tile_number, tile in enumerate(processing_tiles):
            tile.tile_number = tile_number
            tile.output_tile_location = self.output_tile_location
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
            resolution = src.res
            crs = src.crs
            left = src.bounds[0]
            top = src.bounds[3]

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
                tiles.append(Tile((tile_r, tile_c), pos, height, width, 
                                  resolution, crs, left, top))

        return tiles, step_width, step_height

    """def process_tile(self, segmented_img, original_orthomosaic, crd, tile):
        
        width = tile.size[1]
        height = tile.size[0]
        

        crd.main(tile)


        # save results
        tile.ulc_global = [
                self.top - (tile.ulc[0] * self.resolution[0]), 
                self.left + (tile.ulc[1] * self.resolution[1])]
        transform = Affine.translation(
            tile.ulc_global[1] + self.resolution[0] / 2,
            tile.ulc_global[0] - self.resolution[0] / 2) * \
            Affine.scale(self.resolution[0], -self.resolution[0])

        # optional save of results - just lob detection and thresholding result
        #self.save_results(tile.img, tile.tile_number,
        #                  self.resolution, height, width, self.crs, transform)

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
            new_dataset.close()"""