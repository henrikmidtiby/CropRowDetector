import cv2
import rasterio
from rasterio.transform import Affine



class Tile:
    def __init__(self, start_point, position, height, width, 
                 resolution, crs, left, top):
        # Data for the tile
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]

        self.resolution = resolution
        self.crs = crs
        self.left = left
        self.top = top

        self.ulc_global = [
                self.top - (self.ulc[0] * self.resolution[0]), 
                self.left + (self.ulc[1] * self.resolution[1])]
        self.transform = Affine.translation(
            self.ulc_global[1] + self.resolution[0] / 2,
            self.ulc_global[0] - self.resolution[0] / 2) * \
            Affine.scale(self.resolution[0], -self.resolution[0])

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


        #information for the world coordinates
        

    def load_tile_info(self, output_tile_location, generate_debug_images, tile_boundry, threshold_level, expected_crop_row_distance):
        self.output_tile_location = output_tile_location
        self.generate_debug_images = generate_debug_images
        self.tile_boundry = tile_boundry
        self.threshold_level = threshold_level
        self.expected_crop_row_distance = expected_crop_row_distance

    def load_tile_to_world_coordinates(self, resolution, crs, left, top):
        self.resolution = resolution
        self.crs = crs
        self.left = left
        self.top = top
        
        

    def save_tile(self):
        #if self.original_orthomosaic is not None:
        #    print("hej")
        if self.output_tile_location is not None:
            name_mahal_results = f'{ self.output_tile_location }/mahal{ self.tile_number:04d}.tiff'
            img_to_save = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            channels = img_to_save.shape[2]
            temp_to_save = img_to_save.transpose(2, 0, 1) 
            new_dataset = rasterio.open(name_mahal_results,
                                        'w',
                                        driver='GTiff',
                                        res=self.resolution,
                                        height=self.size[0],
                                        width=self.size[1],
                                        count=channels,
                                        dtype=temp_to_save.dtype,
                                        crs=self.crs,
                                        transform=self.transform)
            new_dataset.write(temp_to_save)
            new_dataset.close()