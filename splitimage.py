import cv2
import os
import numpy as np

def split_image(input_image_path, output_folder, tile_size=(2048, 2048)):
    # Read the large .tiff image
    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not read the image.")
        return

    img_height, img_width = image.shape[:2]
    tile_height, tile_width = tile_size

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    # Loop over the image and save tiles
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            # Extract the tile
            tile = image[y:y + tile_height, x:x + tile_width]

            # If the tile is not 2048x2048, pad it with black pixels
            if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                padded_tile = np.zeros((tile_height, tile_width) + tile.shape[2:], dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile

            # Define the filename for the tile
            tile_filename = os.path.join(output_folder, f"tile_{count}.tiff")

            # Save the tile
            cv2.imwrite(tile_filename, tile)
            print(f"Saved {tile_filename}")
            count += 1

input_image_path = '3rd_full.tiff'
output_folder = 'dataset_1:4_3rd'
split_image(input_image_path, output_folder, (2048, 8192))
