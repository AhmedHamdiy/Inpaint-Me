import cv2 as cv
import numpy as np
from .Utils import *



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


        # Update the Confidence s
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


# To Make The Algorithm Run Fast
# pass the stride parameter to be large enough


# Test Cases

# # 1
# first_image = cv.imread("tests/1.png")
# first_mask = cv.imread("masks/1_mask.jpg", cv.IMREAD_GRAYSCALE)
# first_mask = cv.threshold(first_mask, 127, 255, cv.THRESH_BINARY)[1]
# first_mask = first_mask // 255
# first_mask = 1 - first_mask


# 2
# second_image = cv.imread("tests/2.jpg")
# second_mask = generate_rect_mask(second_image, (359, 364), (537, 463))
# second_mask = 1 - second_mask
# cv.imwrite("mask.jpg", second_mask * 255)

# second_experiment = Inpaint(second_image, second_mask, 9)
# second_result = second_experiment.inpaint(2000)
# cv.imwrite("results/second_result.jpg", second_result)  

# # 4
# fourth_image = cv.imread("tests/4.png")
# fourth_mask = generate_rect_mask(fourth_image,(607, 356), (876, 722))
# fourth_mask = 1 - fourth_mask

# #5
# fifth_image = cv.imread("tests/5.jpg")
# fifth_mask = cv.imread("masks/5_mask.jpg", cv.IMREAD_GRAYSCALE)
# fifth_mask = cv.threshold(fifth_mask, 127, 255, cv.THRESH_BINARY)[1]
# fifth_mask = fifth_mask // 255

# #6
# sixth_image = cv.imread("tests/6.jpg")
# sixth_mask = cv.imread("masks/6_mask.jpg", cv.IMREAD_GRAYSCALE)
# sixth_mask = cv.threshold(sixth_mask, 127, 255, cv.THRESH_BINARY)[1]
# sixth_mask = sixth_mask // 255


# #7
# seventh_image = cv.imread("tests/7.jpg")
# seventh_mask = cv.imread("masks/7_mask.jpg", cv.IMREAD_GRAYSCALE)
# seventh_mask = cv.threshold(seventh_mask, 127, 255, cv.THRESH_BINARY)[1]
# seventh_mask = seventh_mask // 255



# #8
# eighth_image = cv.imread("tests/8.jpg")
# eighth_mask = cv.imread("masks/8_mask.jpg", cv.IMREAD_GRAYSCALE)
# eighth_mask = cv.threshold(eighth_mask, 127, 255, cv.THRESH_BINARY)[1]
# eighth_mask = eighth_mask // 255


# images = [first_image, second_image, fourth_image, fifth_image, sixth_image, seventh_image, eighth_image]
# masks = [first_mask, second_mask, fourth_mask, fifth_mask, sixth_mask, seventh_mask, eighth_mask]

# for i in range(len(images)):
#     images[i] = cv.resize(images[i], (256, 256))
#     masks[i] = cv.resize(masks[i], (256, 256))

# first_image, second_image, fourth_image, fifth_image, sixth_image, seventh_image, eighth_image = images
# first_mask, second_mask, fourth_mask, fifth_mask, sixth_mask, seventh_mask, eighth_mask = masks


# cv.imwrite("mask.jpg", second_mask * 255)


#############################





#############################


# print("Test Case Number: ", sys.argv[1])
# sys.argv[1] = int(sys.argv[1])


# # if sys.argv[1] == 1:
# #     first_experiment = Inpaint(first_image, first_mask, 9)
# #     first_result = first_experiment.inpaint(2000)
# #     cv.imwrite("results/first_result.jpg", first_result)
# elif sys.argv[1] == 2:
#     second_experiment = Inpaint(second_image, second_mask, 9)
#     second_result = second_experiment.inpaint(2000)
#     cv.imwrite("results/second_result.jpg", second_result)  
# elif sys.argv[1] == 4:
#     fourth_experiment = Inpaint(fourth_image, fourth_mask, 9, 9)
#     fourth_result = fourth_experiment.inpaint(2000)
#     cv.imwrite("results/fourth_result.jpg", fourth_result)
# elif sys.argv[1] == 5:
#     fifth_experiment = Inpaint(fifth_image, fifth_mask, 5)
#     fifth_result = fifth_experiment.inpaint(2000)
#     cv.imwrite("results/fifth_result.jpg", fifth_result)
# elif sys.argv[1] == 6:
#     sixth_experiment = Inpaint(sixth_image, sixth_mask, 9)
#     sixth_result = sixth_experiment.inpaint(2000)
#     cv.imwrite("results/sixth_result.jpg", sixth_result)
# elif sys.argv[1] == 7:
#     seventh_experiment = Inpaint(seventh_image, seventh_mask, 9)
#     seventh_result = seventh_experiment.inpaint(2000)
#     cv.imwrite("results/seventh_result.jpg", seventh_result)
# elif sys.argv[1] == 8:
#     eighth_experiment = Inpaint(eighth_image, eighth_mask, 9)
#     eighth_result = eighth_experiment.inpaint(2000)
#     cv.imwrite("results/eighth_result.jpg", eighth_result)
# else:
#     print("Invalid Test Case Number")
