import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# Paths to your files
npy_file_path = 'labels_4096/tile_35.npy'
img_file_path = 'labels_4096/tile_35.tiff'
output_file_path = 'labels_4096/tile_35.tiff'

# Load the .npy file and the image
labels = np.load(npy_file_path)
image = plt.imread(img_file_path)

image = np.array(plt.imread(img_file_path))

if image.shape[2] == 4:
    green = np.array([0, 255, 0, 255])
    red = np.array([255, 0, 0, 255])
    black = np.array([0, 0, 0, 255])
else:
    green = np.array([0, 255, 0])
    red = np.array([255, 0, 0])
    black = np.array([0, 0, 0])

colordict = {}
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        if labels[y, x] not in colordict:
            colordict[labels[y, x]] = image[y, x]
        elif not np.array_equal(image[y, x], black):
            colordict[labels[y, x]] = image[y, x]
        


# Toggle the color of a segment
def toggle_segment_colors(segment_ids):
    for s in segment_ids:
        col = colordict[s]
        if np.array_equal(col, black):
            continue
        if np.array_equal(col, red):
            colordict[s] = green
        else:
            colordict[s] = red
def colorsegs(segment_ids, color):
    for s in segment_ids:
        if np.array_equal(colordict[s], black):
            continue
        colordict[s] = color
        
def update_image(x1, y1, x2, y2, segs, width = 50):
    s = set(segs)
    x1 = max(0, x1-width)
    y1 = max(0, y1-width)
    x2 = min(image.shape[1], x2+width)
    y2 = min(image.shape[0], y2+width)
    for y in range(y1, y2):
        for x in range(x1, x2):
            if not np.array_equal(image[y, x], black) and labels[y, x] in s:
                image[y, x] = colordict[labels[y, x]]
    

# Display the image with segment borders
def display_image(labels, image):
    fig, ax = plt.subplots(figsize=(15, 15))  # Set the window size here

    ax.imshow(image)

    dragging = False
    start_x, start_y = 0, 0
    end_x, end_y = 0, 0
    rect = None

    def on_click(event):
        nonlocal dragging, start_x, start_y, rect
        if event.button == MouseButton.LEFT and event.inaxes:
            start_x, start_y = int(event.xdata), int(event.ydata)
            dragging = True
           
            

    def on_release(event):
        nonlocal dragging, end_x, end_y, rect
        if event.button == MouseButton.LEFT and dragging:
            end_x, end_y = int(event.xdata), int(event.ydata)
            rect = plt.Rectangle((start_x, start_y), end_x-start_x, end_y-start_y, fill=False, edgecolor='blue', linestyle='--')
            ax.add_patch(rect)
            dragging = False
            plt.draw()


    def on_key(event):
        nonlocal start_x, start_y, end_x, end_y, rect
        if event.key == 't':
            segs = np.unique(labels[start_y:end_y, start_x:end_x])
            toggle_segment_colors(np.unique(labels[start_y:end_y, start_x:end_x]))
            rect.remove()  # Remove the rectangle
            update_image(start_x, start_y, end_x, end_y, segs)
            ax.clear()
            ax.imshow(image)
            plt.draw()
        if event.key == 'g':
            segs = np.unique(labels[start_y:end_y, start_x:end_x])
            colorsegs(np.unique(labels[start_y:end_y, start_x:end_x]), green)
            rect.remove()  # Remove the rectangle
            update_image(start_x, start_y, end_x, end_y, segs)
            ax.clear()
            ax.imshow(image)
            plt.draw()
        if event.key == 'r':
            segs = np.unique(labels[start_y:end_y, start_x:end_x])
            colorsegs(np.unique(labels[start_y:end_y, start_x:end_x]), red)
            rect.remove()  # Remove the rectangle
            update_image(start_x, start_y, end_x, end_y, segs)
            ax.clear()
            ax.imshow(image)
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    def on_close(event):
        plt.imsave(output_file_path, image)
        print(f"Image saved to {output_file_path}")

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

# Display the image
display_image(labels, image)