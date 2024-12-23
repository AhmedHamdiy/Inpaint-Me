import cv2 as cv
import numpy as np
from Utils import *



class Inpaint:

    def calculate_matrices(self):

        contours = get_contours(self.mask * 255)

        if len(contours) == 0:
            self.finish = True
            return
        
        self.contour = contours[0]
        self.normals = get_normals(self.contour) 

    def initalize_confidence(self):
        self.confidence = np.zeros((self.height, self.width))
        self.confidence[self.mask == 0] = 1
    
    def update_confidence(self, point, value):
        confidence_patch = get_patch(point, self.confidence, self.patch_size)
        confidence_patch[:] = value

    def range_patchs(self):
        h = w = self.patch_size

        for y in range(0, self.height - h + 1, self.stride):
            for x in range(0, self.width - w + 1):

                patch = self.img[y:y+h, x:x+w]
                patch_mask = self.mask[y:y+h, x:x+w]

                if np.any(patch_mask == 1):
                    continue

                yield patch, x, y

    def get_patchs(self):
        for patch, x, y in self.range_patchs():
            self.patchs.append([patch, x, y])
        
    def calculate_confidence(self):

        contour_confidence = []

        for point in self.contour:
            patch = get_patch(point, self.confidence, self.patch_size)
            contour_confidence.append(np.mean(patch))

        return contour_confidence
    
    def calculate_data(self):

        contour_data = []

        for i in range(len(self.contour)):

            point = self.contour[i]
            normal = self.normals[i]

            gradient = calculate_gradient(self.img, point)
            isophate = np.array([gradient[1], -gradient[0]])
            isophate = isophate / (np.linalg.norm(isophate) + 1e-8)

            contour_data.append(np.abs(np.dot(isophate, normal)))

        return contour_data

    def get_best_exemplar(self, target_patch, mask_patch_3d):

        min_distance = np.inf
        min_x, min_y = -1, -1
        dist_x, dist_y = 0, 0

        for patch, x, y in self.patchs:

            patch = patch[:target_patch.shape[0], :target_patch.shape[1], :]

            diff = target_patch - patch
            diff = diff * (1 - mask_patch_3d)

            distance = np.sum(diff ** 2)

            if distance < min_distance:
                min_distance = distance
                min_x, min_y = x, y
                dist_x, dist_y = target_patch.shape[1], target_patch.shape[0]

        return (min_x, min_y), (dist_x, dist_y)
        

    def __init__(self, img, mask, patch_size = -1, stride = 1):
        print("Inpainting")

        self.stride = stride 
        self.finish = False

        if(patch_size == -1):
            self.patch_size = compute_texel_size(img)
        else:
            self.patch_size = patch_size

        self.height, self.width = img.shape[:2]
        self.img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.mask = mask
        self.patchs = []

        self.calculate_matrices()
        self.initalize_confidence()
        self.get_patchs()



    
    def inpaint_iter(self):

        if(self.finish):
            return
        
        contour_confidence = self.calculate_confidence()
        contour_data = self.calculate_data()
        contour_priority = np.array(contour_confidence) * np.array(contour_data)


        target_point_index = np.argmax(contour_priority)
        target_point = self.contour[target_point_index]
        target_patch = get_patch(target_point, self.img, self.patch_size)
        mask_patch = get_patch(target_point, self.mask, self.patch_size)
        mask_patch_3d = cv.cvtColor(mask_patch, cv.COLOR_GRAY2RGB)        

        (x, y), (dist_x, dist_y) = self.get_best_exemplar(target_patch, mask_patch_3d)

        best_patch = self.img[y:y+dist_y, x:x+dist_x, :]       
        target_patch[:] = best_patch * mask_patch_3d + target_patch * (1 - mask_patch_3d) # Inpainting
        mask_patch[:] = 0 # Update the mask


        self.update_confidence(target_point, contour_confidence[target_point_index])
        self.calculate_matrices()

        self.iterations -= 1

    def inpaint(self, iters = 1000):

        self.iterations = iters

        for _ in range(self.iterations):
            print(f"Iteration: {iters - self.iterations}")
            self.inpaint_iter()

            if self.finish:
                break

        
        return cv.cvtColor(self.img, cv.COLOR_RGB2BGR).copy()
