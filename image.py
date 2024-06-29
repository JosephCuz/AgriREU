import cv2
import numpy as np
from fast_slic import Slic



def correct_barrel_distortion(image, fx, fy, cx, cy, dist_coeffs):
    # Load the image

    # Define the camera matrix (intrinsic parameters)
    h, w = image.shape[:2]
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Get the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on the ROI (Region of Interest)
    x, y, w, h = roi
    return undistorted_image[y:y+h, x:x+w]

def mask_image(image):
    B, G, R = cv2.split(image)
    B_norm, G_norm, R_norm = B/255, G/255, R/255
    denom = cv2.normalize(B_norm + G_norm + R_norm, None, 0.0001, 3, cv2.NORM_MINMAX)

    b, g, r = B_norm / denom, G_norm / denom, R_norm / denom

    # Compute the ExG index
    ExG = 2 * g - r - b
    #GLI = (2 * G - R - B) / (2 * G + R + B)

    # Normalize the ExG index to the range 0-255 for display purposes
    ExG_normalized = cv2.normalize(ExG, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.jpg', ExG_normalized)

    # Convert the ExG image to uint8 type
    ExG_normalized = ExG_normalized.astype(np.uint8)

    # Apply a higher global threshold
    _, high_thresh = cv2.threshold(ExG_normalized, 100, 255, cv2.THRESH_BINARY)

    return high_thresh

def drawlines(image, lines, col = (255, 0, 0)):
    res = image.copy()
    for coord in lines:
        res = cv2.line(res, (coord[0][0] ,coord[0][1]), (coord[1][0], coord[1][1]), col, 2)
    return res

def slic(image, segments):
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    slic = Slic(num_components=segments, compactness=20)

# Apply SLIC algorithm
    segments = slic.iterate(image_lab, 10) # Cluster Map
    

# Convert the segmented image back to RGB
    return segments

def draw_superpixel_borders(image, labels):
    # Convert labels to a format suitable for boundary detection    
    # Create an empty image to draw the borders on
    borders = np.zeros(image.shape[:2], dtype=np.uint8)

    # Iterate over each pixel and check if its neighbor has a different label
    for y in range(1, labels.shape[0] - 1):
        for x in range(1, labels.shape[1] - 1):
            if (labels[y, x] != labels[y, x + 1] or 
                labels[y, x] != labels[y + 1, x]):
                borders[y, x] = 255

    # Dilate the borders for better visibility
    #borders = cv2.dilate(borders, None)

    # Create an output image to draw the borders
    output = image.copy()
    output[borders == 255] = [0, 255, 0]  # Green borders

    return output

def mask_superpixels_intersecting_line(image, labels, lines):
    # Convert labels to a format suitable for processing
    labels = labels.astype(np.int32)

    # Create a mask image to draw the line
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = drawlines(mask, lines, 255)
    # Find unique labels that intersect the line
    intersecting_labels = np.unique(labels[mask == 255])

    # Create a mask for the intersecting superpixels
    superpixel_mask = np.zeros_like(mask)
    for label in intersecting_labels:
        superpixel_mask[labels == label] = 255

    # Apply the mask to the original image (masking superpixels)
    masked_image = image.copy()
    masked_image[superpixel_mask != 255] = [0, 0, 0]  # Black out the nonintersecting superpixels

    return masked_image, superpixel_mask

def hough_transform(skeleton, anglethr = 0.1, angleoffset = 0.05, houghthr = 750):
    h, w = skeleton.shape[:2]
    lines = cv2.HoughLines(skeleton, 1, np.pi / 1200, houghthr, None, min_theta=np.pi/2+angleoffset - anglethr, max_theta=np.pi/2+angleoffset + anglethr)
    linesxy = []
 
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - (w+500)*(-b)), int(y0 - (h+500)*(a)))
            pt2 = (int(x0 + w*(-b)), int(y0 + h*(a)))
            linesxy.append((pt1, pt2))
    return linesxy

def hough_transformP(skeleton, anglethr = 0.1, angleoffset = 0.05, houghthr = 300):
    h, w = skeleton.shape[:2]
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 1200, houghthr, None)
    
    return [[(line[0][0], line[0][1]), (line[0][2], line[0][3])] for line in lines]

if __name__ == "__main__":

    print('loading')
    #image = cv2.imread('dataset_2048/tile_18432_14336.tiff')
    image = cv2.imread('dataset_4096/tile_19.tiff')

    h, w = image.shape[:2]
    fx, fy = w, h  # Approximation, should be calibrated for better results
    cx, cy = w / 2, h / 2

    #undistorted = correct_barrel_distortion(image, fx, fy, cx, cy, np.array([-0.15, 0.03, 0, 0, 0])) 
    #cropped = undistorted[:, 0:w-800]

    #cv2.imwrite('undistorted.jpg', cropped)
    print('masking')
    mask = mask_image(image)
    print('skeletonizing')
    sk = cv2.ximgproc.thinning(mask)
    #sk = cv2.Canny(mask, 50, 200, 3)

    print('Hough transform')
    linesxy = hough_transformP(sk, houghthr=700)

    lines = drawlines(image, linesxy)
    print('slic')
    segments = slic(image, 10000)
    print('superpixel borders')
    superpixels = draw_superpixel_borders(image, segments)
    superpixels = drawlines(superpixels, linesxy)
    cv2.imwrite('test.tif', superpixels)
    print('intersecting')
    intersections, sp_mask = mask_superpixels_intersecting_line(image, segments, linesxy)

    print('finishing')
    crop_mask = cv2.bitwise_and(mask, sp_mask)
    weed_mask = cv2.bitwise_and(mask, cv2.bitwise_not(sp_mask))
    plantseg = np.zeros_like(image)
    plantseg[crop_mask == 255] = (0, 255, 0)
    plantseg[weed_mask == 255] = (0, 0, 255)


    cv2.imwrite('plantseg.jpg', plantseg)
    cv2.imwrite('cropmask.jpg', crop_mask)
    cv2.imwrite('weedmask.jpg', weed_mask)



