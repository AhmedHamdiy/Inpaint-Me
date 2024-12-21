import cv2 as cv
import numpy as np

# Height * Width * 3
# Mask Calculation : Checked 


class Inpaint:

    @staticmethod
    def get_patch(center_pixel, img, patch_size):

        half_size = patch_size // 2

        center_x, center_y = center_pixel

        start_x = max(center_x - half_size, 0)
        end_x = min(center_x + half_size + 1, img.shape[0])

        start_y = max(center_y - half_size, 0)
        end_y = min(center_y + half_size + 1, img.shape[1])

        return img[start_x:end_x, start_y:end_y]

    @staticmethod
    def computeContours(image):
        contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        filtered_contours = []

        for contour in contours:
            new_contour = []    
            for point in contour:
                point = point[0]  # Extract the (x, y) coordinates
            
                # Skip points on the image borders
                if point[0] == 0 or point[1] == 0:
                    continue
                if point[0] == image.shape[1] - 1 or point[1] == image.shape[0] - 1:
                    continue
                
                new_contour.append(point)
            
            if new_contour:
                filtered_contours.append(np.array(new_contour, dtype=np.int32))

        return filtered_contours

    @staticmethod    
    def computeNormals(contour):

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

    def __init__(self, img, mask, patch_size):

        self.patch_size = patch_size
        self.height, self.width = img.shape[:2]

        self.img = img
        self.grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.mask = (mask > 0).astype(np.uint8)
        self.rgb_img[self.mask == 0] = 0

        self.contour = Inpaint.computeContours(self.mask)[0]
 
        self.normals = Inpaint.computeNormals(self.contour)

        self.confidence = np.zeros((self.height, self.width))
        self.confidence[self.mask == 1] = 1

        self.gradientX = cv.Sobel(self.grey_img, cv.CV_64F, 1, 0, ksize=3)
        self.gradientY = cv.Sobel(self.grey_img, cv.CV_64F, 0, 1, ksize=3)


    def patchs(self):

        h = w = self.patch_size

        for y in range(0, self.height - h + 1):
            for x in range(0, self.width - w + 1):
                
                patch = self.rgb_img[y:y+h, x:x+w]
                patch_mask = self.mask[y:y+h, x:x+w]

                if np.any(patch_mask == 0):
                    continue

                yield patch, x, y



    def calculateConfidence(self):

        contour_confidence = []

        for point in self.contour:
            patch = Inpaint.get_patch(point, self.confidence, self.patch_size)
            contour_confidence.append(np.mean(patch))

        return contour_confidence
    
    def calculateData(self):

        contour_data = []

        for i in range(len(self.contour)):

            point = self.contour[i]
            normal = self.normals[i]

            gradient = np.array([self.gradientX[point[1], point[0]], self.gradientY[point[1], point[0]]])
            isophate = np.array([gradient[1], -gradient[0]])
            isophate = isophate / (np.linalg.norm(isophate) + 1e-8)

            contour_data.append(np.dot(isophate, normal))

        return contour_data
    
    def update(self):

        self.contour = Inpaint.computeContours(self.mask)[0]
        self.normals = Inpaint.computeNormals(self.contour)

        self.grey_img = cv.cvtColor(self.rgb_img, cv.COLOR_BGR2GRAY)

        self.gradientX = cv.Sobel(self.grey_img, cv.CV_64F, 1, 0, ksize=3)
        self.gradientY = cv.Sobel(self.grey_img, cv.CV_64F, 0, 1, ksize=3)



    # Single inpainting iteration
    def inpaint(self):
        contour_confidence = self.calculateConfidence()
        contour_data = self.calculateData()

        contour_priority = np.array(contour_confidence) * np.array(contour_data)
        max_index = np.argmax(contour_priority)

        point = self.contour[max_index]

        target_patch = Inpaint.get_patch(point, self.rgb_img, self.patch_size)
        mask_patch = Inpaint.get_patch(point, self.mask, self.patch_size)
        mask_patch_expanded = mask_patch[:, :, np.newaxis]

        if(target_patch.shape[0] != self.patch_size or target_patch.shape[1] != self.patch_size):
            self.contour = np.delete(self.contour, max_index, 0)
            self.normals = np.delete(self.normals, max_index, 0)
            return

        min_distance = np.inf
        min_x, min_y = -1, -1

        for patch, x, y in self.patchs():
            diff = target_patch - patch
            diff = diff * mask_patch_expanded

            distance = np.sum(diff ** 2)

            if distance < min_distance:
                min_distance = distance
                min_x, min_y = x, y

        rgb_img_prev = self.rgb_img.copy()

        
        best_patch = self.rgb_img[min_y:min_y+self.patch_size, min_x:min_x+self.patch_size, :]
        target_patch_prev = target_patch.copy()
        target_patch[:] = best_patch * (1 - mask_patch_expanded) + target_patch * mask_patch_expanded
        self.rgb_img[min_y:min_y+self.patch_size, min_x:min_x+self.patch_size, :] = target_patch

        mask_patch[:] = 1

        # Update the Confidence s
        confidence_patch = Inpaint.get_patch(point, self.confidence, self.patch_size)
        confidence_patch[:] = contour_confidence[max_index]

        if(np.all(rgb_img_prev == self.rgb_img)):
            print(self.iterations)
            print('No change')
    

        self.update()
        self.iterations -= 1

    def inpaint_all(self, iterations):
        self.iterations = iterations

        while(self.iterations > 0):
            self.inpaint()
        
        return self.rgb_img

    



image = cv.imread('images/1.png')

# create a circle mask in the center of the image
mask = np.zeros(image.shape[:2], np.uint8)

# Calculate the center of the image
center = (image.shape[1] // 2, image.shape[0] // 2)

# Create a circle at the center
mask = cv.circle(mask, center, 50, 255, -1)
mask = np.invert(mask)
mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)[1]

mask[mask == 255] = 1


inpaint = Inpaint(image, mask, 11)

result = inpaint.inpaint_all(100)

diff = cv.absdiff(result, image)



cv.imwrite('images/1_inpaint.png', diff)

# result = inpaint.inpaint_all(1000)

# cv.imwrite('images/1_inpaint.png', result)








        


    

        





        

            

        

            




















        

        

   







    





        







