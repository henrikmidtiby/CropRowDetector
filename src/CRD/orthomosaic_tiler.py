"""Tile Orthomosaics into smaller pieces for easier processing."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.enums import Resampling
from rasterio.windows import Window


class Tile:
    """
    Handle all information of a tile with read and write.

    Parameters
    ----------
    orthomosaic
        The orthomosaic from where the tile is taken.
    Upper_left_corner
        The pixel coordinate from the orthomosaic of the upper left corner of the tile in (columns, rows).
    position
        Tile position in orthomosaic in number of tile in (columns, rows).
    width
        Tile width.
    height
        Tile height.
    overlap
        Overlap in percentage of width and height.
    number
    """

    def __init__(
        self,
        orthomosaic: pathlib.Path,
        Upper_left_corner: tuple[int, int],
        position: tuple[int, int],
        width: float,
        height: float,
        overlap: float = 0.01,
        number: int = 0,
    ):
        # Data for the tile
        self.orthomosaic = orthomosaic
        self.size = (width, height)
        self.tile_position = position
        self.ulc = Upper_left_corner
        self.overlap = overlap
        self.tile_number = number
        """The tile number."""
        self.output: NDArray[Any] = np.zeros(0)
        """np.ndarray : processed output of tile to save for later use."""
        self.set_tile_data_from_orthomosaic()

    def set_tile_data_from_orthomosaic(self) -> None:
        """Read data about the tile from the orthomosaic."""
        try:
            with rasterio.open(self.orthomosaic) as src:
                self.ortho_cols = src.width
                self.ortho_rows = src.height
                self.resolution = src.res
                self.crs = src.crs
                left = src.bounds[0]
                top = src.bounds[3]
                transform = src.transform
        except rasterio.RasterioIOError as e:
            raise OSError(f"Could not open the orthomosaic at '{self.orthomosaic}'") from e
        self.ulc_global = [
            left + (self.ulc[0] * self.resolution[0]),
            top - (self.ulc[1] * self.resolution[1]),
        ]
        self._set_window_with_overlap()
        self.transform = rasterio.windows.transform(self.window, transform)

    def _set_window_with_overlap(self):
        pixel_overlap_width = int(self.size[0] * self.overlap)
        pixel_overlap_hight = int(self.size[1] * self.overlap)
        start_col = self.ulc[0] - pixel_overlap_width
        stop_col = self.ulc[0] + self.size[0] + pixel_overlap_width
        start_row = self.ulc[1] - pixel_overlap_hight
        stop_row = self.ulc[1] + self.size[1] + pixel_overlap_hight
        if start_col < 0:
            start_col = 0
        if stop_col > self.ortho_cols:
            stop_col = self.ortho_cols
        if start_row < 0:
            start_row = 0
        if stop_row > self.ortho_rows:
            stop_row = self.ortho_rows
        self.window = Window.from_slices(
            (start_row, stop_row),
            (start_col, stop_col),
        )

    def read_tile(self) -> NDArray[Any]:
        """Read the tiles image data from the orthomosaic."""
        with rasterio.open(self.orthomosaic) as src:
            img: NDArray[Any] = src.read(window=self.window)
            mask = src.read_masks(window=self.window)
            self.mask = mask[0]
            for band in range(mask.shape[0]):
                self.mask = self.mask & mask[band]
        return img

    def save_tile(self, image: NDArray[Any], output_tile_location: pathlib.Path) -> None:
        """Save the image of the tile to a tiff file. Filename is the tile number."""
        self.output = image
        if not output_tile_location.is_dir():
            os.makedirs(output_tile_location)
        output_tile_filename = output_tile_location.joinpath(f"{self.tile_number:05d}.tiff")
        with rasterio.open(
            output_tile_filename,
            "w",
            driver="GTiff",
            res=self.resolution,
            width=image.shape[1],
            height=image.shape[2],
            count=image.shape[0],
            dtype=image.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as new_dataset:
            new_dataset.write(image)
            if image.shape[0] == 1:
                new_dataset.write_mask(self.mask)


class OrthomosaicTiles:
    """
    Convert orthomosaic into tiles.

    Parameters
    ----------
    orthomosaic
    tile_size
        tile size in pixels.
    overlap
        How much the tiles should overlap in percentage of the tile size.
    run_specific_tile
        List of tiles to run e.g. [15, 65] runs tiles 15 and 65.
    run_specific_tileset
        List of ranges of tiles to run e.g. [15, 65] runs all tiles between 15 and 65.
    """

    def __init__(
        self,
        *,
        orthomosaic: pathlib.Path,
        tile_size: int,
        overlap: float = 0,
        run_specific_tile: list[int] | None = None,
        run_specific_tileset: list[int] | None = None,
    ):
        self.orthomosaic = orthomosaic
        self.tile_size = tile_size
        self.overlap = overlap
        self.run_specific_tile = run_specific_tile
        self.run_specific_tileset = run_specific_tileset
        self.tiles: list[Tile] = []
        """List of tiles"""

    def divide_orthomosaic_into_tiles(self) -> list[Tile]:
        """Divide orthomosaic into tiles and select specific tiles if desired."""
        tiles = self.get_tiles()
        specified_tiles = self.get_list_of_specified_tiles(tiles)
        self.tiles = specified_tiles
        return specified_tiles

    def get_list_of_specified_tiles(self, tile_list: list[Tile]) -> list[Tile]:
        """From a list of all tiles select only specified tiles."""
        specified_tiles = []
        if self.run_specific_tile is None and self.run_specific_tileset is None:
            return tile_list
        if self.run_specific_tile is not None:
            for tile_number in self.run_specific_tile:
                specified_tiles.append(tile_list[tile_number])
        if self.run_specific_tileset is not None:
            for start, end in zip(self.run_specific_tileset[::2], self.run_specific_tileset[1::2], strict=True):
                if start > end:
                    raise ValueError(f"Specific tileset range is negative: from {start} to {end}")
                for tile_number in range(start, end + 1):
                    specified_tiles.append(tile_list[tile_number])
        return specified_tiles

    def get_orthomosaic_size(self) -> tuple[int, int]:
        """
        Read size from orthomosaic.

        Returns
        -------
        columns : int
        rows : int
        """
        try:
            with rasterio.open(self.orthomosaic) as src:
                columns = src.width
                rows = src.height
        except rasterio.RasterioIOError as e:
            raise OSError(f"Could not open the orthomosaic at '{self.orthomosaic}'") from e
        return columns, rows

    def get_tiles(self) -> list[Tile]:
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.

        Returns
        -------
        list of tiles : list[Tile]
        """
        columns, rows = self.get_orthomosaic_size()
        n_height = np.ceil(rows / self.tile_size).astype(int)
        n_width = np.ceil(columns / self.tile_size).astype(int)
        tiles = []
        for r in range(0, n_height):
            for c in range(0, n_width):
                pos = (c, r)
                number = r * n_width + c
                tile_c = c * self.tile_size
                tile_r = r * self.tile_size
                tiles.append(
                    Tile(
                        self.orthomosaic,
                        (tile_c, tile_r),
                        pos,
                        self.tile_size,
                        self.tile_size,
                        self.overlap,
                        number,
                    )
                )
        return tiles

    def save_orthomosaic_from_tile_output(self, orthomosaic_filename: pathlib.Path) -> None:
        """Save an orthomosaic from the processed tiles."""
        if not orthomosaic_filename.parent.exists():
            orthomosaic_filename.parent.mkdir(parents=True)
        output_count = self.tiles[0].output.shape[0]
        with rasterio.open(self.orthomosaic) as src:
            profile = src.profile
            profile["count"] = output_count
            overview_factors = src.overviews(src.indexes[0])
        with rasterio.open(orthomosaic_filename, "w", **profile) as dst:
            for tile in self.tiles:
                dst.write(tile.output, window=tile.window)
                if output_count == 1:
                    dst.write_mask(tile.mask, window=tile.window)
        with rasterio.open(orthomosaic_filename, "r+") as dst:
            dst.build_overviews(overview_factors, Resampling.average)
