import cv2
import os

def split_image(image, tile_size):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
    return tiles

def save_tiles(tiles, output_folder, base_filename):
    for index, tile in enumerate(tiles):
        tile_filename = f"{base_filename}_{index}.tiff"
        cv2.imwrite(os.path.join(output_folder, tile_filename), tile)

def process_folders(folder1, folder2, output_folder1, output_folder2, tile_size=512):
    files1 = sorted(os.listdir(folder1))[1:] # because of .DS_Store
    files2 = sorted(list(filter(lambda x: '.tiff' in x ,os.listdir(folder2))))
    
    for file1, file2 in zip(files1, files2):
        base_filename1 = os.path.splitext(file1)[0]
        base_filename2 = os.path.splitext(file2)[0]

        image1 = cv2.imread(os.path.join(folder1, file1))
        image2 = cv2.imread(os.path.join(folder2, file2))

        tiles1 = split_image(image1, tile_size)
        tiles2 = split_image(image2, tile_size)

        save_tiles(tiles1, output_folder1, base_filename1)
        save_tiles(tiles2, output_folder2, base_filename2)

# Define your folder paths
folder1 = 'dataset_4096'
folder2 = 'labels_4096'
output_folder1 = 'dataset_512_3rd/inputs'
output_folder2 = 'dataset_512_3rd/labels'

process_folders(folder1, folder2, output_folder1, output_folder2)