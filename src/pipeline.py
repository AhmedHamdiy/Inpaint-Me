import cv2
import os
from inpaint import Inpaint
from skimage.transform import resize

def show_image(img, filename):
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(filename, img)

def generate_mask(drawing_path):
    
    image = cv2.imread(drawing_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    mask = binary_mask // 255

    return mask

def pipeline(img, drawing_path, output_dir):

    mask = generate_mask(drawing_path)
    
    resized_img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        
    inpaint_algo = Inpaint(resized_img, mask, 9, 3)
    inpainted_img = inpaint_algo.inpaint(1000)
    
    show_image(inpainted_img, os.path.join(output_dir, "final_result.jpg"))
