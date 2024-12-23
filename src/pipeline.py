import cv2
import numpy as np
import os
from src.inpaint import Inpaint

def show_image(img, filename):
    # Ensure the output directory exists before saving
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(filename, img)

def object_detection(img, kernel_size=(15, 15), distance_scale=0.625, debug=False):
    if img is None:
        raise ValueError("Input image is None.")
    
    original = img.copy()  # Save original for overlaying results later
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, distance_scale * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Identify unknown regions
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(img, markers)

    # Draw contours and bounding boxes
    labels = np.unique(markers)
    objects = []
    for label in labels:
        if label <= 1:  # Skip background and border markers
            continue
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            objects.append(contours[0])

    # Draw detected objects
    img = cv2.drawContours(original, objects, -1, (0, 255, 0), thickness=1)

    # Binaraize the image for filling
    binary = np.zeros_like(img[:,:,0])

    for (x, y, _) in np.argwhere(img==(0,255,0)):
        binary[x, y] = 255

    # Debug visualization
    if debug:
        show_image(thresh, "Thresholded Image")
        show_image(opening, "Morphological Opening")
        show_image(dist_transform.astype(np.uint8), "Distance Transform")
        show_image(sure_fg, "Sure Foreground")
    return img, binary

def fill_region_and_export(binary_image, x, y, output_dir):
    """
    Fills the region inside the contour where the (x, y) point lies and exports the result.
    
    Args:
        binary_image (np.ndarray): The binary image containing the contours.
        x (int): X-coordinate of the selected point.
        y (int): Y-coordinate of the selected point.
        output_dir (str): Directory to save the output images.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Highlight the point on the input image
    image_with_point = cv2.circle(binary_image.copy(), (x, y), radius=3, color=128, thickness=-1)
    show_image(image_with_point, os.path.join(output_dir, "selected_point.png"))

    # Ensure the input is a binary image
    if len(binary_image.shape) != 2 or binary_image.dtype != np.uint8:
        raise ValueError("Input must be a binary image (2D array of dtype uint8).")

    # Check if the given pixel is inside a contour
    if binary_image[y, x] == 255:
        raise ValueError("The given point is inside a contour")

    # Find all contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour containing the point
    contour_found = False
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), measureDist=False) >= 0:
            contour_found = True
            # Create a mask for the specific contour
            mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            show_image(mask, os.path.join(output_dir, "filled_region.png"))
            return mask

    if not contour_found:
        raise ValueError("No contour found containing the given point.")

def get_region_fill_sample():
    """
    Create a complex binary image with a black background and white contours.
    """
    binary_image = np.zeros((400, 400), dtype=np.uint8)

    # Draw multiple white contours for complexity
    cv2.rectangle(binary_image, (50, 50), (150, 150), 255, thickness=2)
    cv2.circle(binary_image, (200, 200), 50, 255, thickness=2)
    cv2.ellipse(binary_image, (300, 300), (60, 40), 0, 0, 360, 255, thickness=2)
    cv2.line(binary_image, (50, 300), (150, 350), 255, thickness=2)
    return binary_image

def start(img, x, y, output_dir):
    """
    Main function to run the object detection and region fill algorithm.
    Args:
        img (np.ndarray): The original colorful image.
        x (int): The x-coordinate of the point inside the image.
        y (int): The y-coordinate of the point inside the image.
        output_dir (str): Directory to save the output images.
    """
    result, binary = object_detection(img, debug=False)
    show_image(result, os.path.join(output_dir, "detected_objects.png"))
    show_image(binary, os.path.join(output_dir, "binary_image.png"))

    # Call the function to fill the region inside the contour
    mask = fill_region_and_export(binary, x, y, output_dir)
    
    inpaint_algo = Inpaint(img, mask, 9)
    inpainted_img = inpaint_algo.inpaint(1000)
    
    inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)

    show_image(inpainted_img, os.path.join(output_dir, "final_result.jpg"))

def show_image(img, filename):
    """
    Utility to show and save an image to the specified path.
    
    Args:
        img (np.ndarray): Image to show and save.
        filename (str): Path where to save the image.
    """
    if img is not None:
        cv2.imwrite(filename, img)
