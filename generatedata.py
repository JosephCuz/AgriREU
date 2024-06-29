from image import *
import os

def process(path):
    print('preprocessing')

    image = cv2.imread(path)

    h, w = image.shape[:2]
    
    mask = mask_image(image)
    sk = cv2.ximgproc.thinning(mask)

    linesxy = hough_transform(sk)
    print('slic')
    segments = slic(image, 10000)
    print('intersecting')
    intersections, sp_mask = mask_superpixels_intersecting_line(image, segments, linesxy)


    crop_mask = cv2.bitwise_and(mask, sp_mask)
    weed_mask = cv2.bitwise_and(mask, cv2.bitwise_not(sp_mask))
    img = np.zeros_like(image)
    #img = plantseg.copy()
    # plantseg[crop_mask == 255] = 1
    # plantseg[weed_mask == 255] = 2
    img[crop_mask == 255] = (0, 255, 0)
    img[weed_mask == 255] = (0, 0, 255)
    
    #np.save('labels_4096_2nd' + path[12:-5], plantseg)
    np.save('labels_4096' + path[12:-5] + '.npy', segments)
    cv2.imwrite('labels_4096' + path[12:-5] + '.tiff', img)




if __name__ == "__main__":
    for filename in os.listdir('dataset_4096'):
        f = os.path.join('dataset_4096', filename)
        if os.path.isfile(f) and 'Store' not in f:
            print(f)
            process(f)
            
