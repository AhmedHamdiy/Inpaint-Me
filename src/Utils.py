import numpy as np
import cv2 as cv


def compute_texel_size(img):
    """
    Compute the texel size based on image size
    (make sure it's odd so that the kernel has a well-defined center).

    Args:
        img (np.ndarray): The input image.

    Returns:
        int: The size of the texel
    """
    texel_size = min(img.shape[:-1]) // 30

    # Ensure texel size is odd
    if texel_size % 2 == 0:
        texel_size += 1
    return texel_size

def calculate_gradient(img, point):

    x, y = point

    height, width = img.shape[:2]

    first_x = min(x + 1, width - 1)
    second_x = max(0, x - 1)

    first_y = min(y + 1, height - 1)
    second_y = max(0, y - 1)

    x1 = np.float32(img[y, first_x, 0]) + np.float32(img[y, first_x, 1]) + np.float32(img[y, first_x, 2])
    x2 = np.float32(img[y, second_x, 0]) + np.float32(img[y, second_x, 1]) + np.float32(img[y, second_x, 2])
    y1 = np.float32(img[first_y, first_x, 0]) + np.float32(img[first_y, first_x, 1]) + np.float32(img[first_y, first_x, 2])
    y2 = np.float32(img[second_y, first_x, 0]) + np.float32(img[second_y, first_x, 1]) + np.float32(img[second_y, first_x, 2])
    
    x1 = np.clip(x1 // 3, 0, 255)
    x2 = np.clip(x2 // 3, 0, 255)
    y1 = np.clip(y1 // 3, 0, 255)
    y2 = np.clip(y2 // 3, 0, 255)

    dx = (x1 - x2) / 2
    dy = (y1 - y2) / 2

    return dx, dy


def generate_rect_mask(img, point1, point2):

    height, width = img.shape[:2]

    x1 = max(0, min(point1[0], width - 1))
    x2 = max(0, min(point2[0], width))
    y1 = max(0, min(point1[1], height - 1))
    y2 = max(0, min(point2[1], height))

    mask = np.ones((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 0

    return mask


def get_patch(center_pixel, img, patch_size):

    half_size = patch_size // 2

    center_x, center_y = center_pixel

    start_x = max(center_x - half_size, 0)
    end_x = min(center_x + half_size + 1, img.shape[1])

    start_y = max(center_y - half_size, 0)
    end_y = min(center_y + half_size + 1, img.shape[0])

    return img[start_y:end_y, start_x:end_x]


def get_contours(image_):

    image = image_ * 255
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    filtered_contours = []

    for contour in contours:
        new_contour = []    
        for point in contour:
            point = point[0]  
        
            if point[0] == 0 or point[1] == 0:
                continue
            if point[0] == image.shape[1] - 1 or point[1] == image.shape[0] - 1:
                continue
            
            new_contour.append(point)
        
        if new_contour:
            filtered_contours.append(np.array(new_contour, dtype=np.int32))

    return filtered_contours


def get_normals(contour):

    normals = []

    for i in range(len(contour)):
        next_point = contour[(i + 1) % len(contour)]
        prev_point = contour[i - 1]

        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]

        normal = np.array([-dy, dx])
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        normals.append(normal)

    return normals


