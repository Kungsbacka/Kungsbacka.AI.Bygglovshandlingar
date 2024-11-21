
import response_model as rm
import cv2
import numpy as np



def validate_situationsplan(image):
    """
    Inputs
        image: np.array
    Returns
        bool: True if situationsplan is valid, False otherwise
    """
    template_path = "template_shild.png"
    template = cv2.imread(template_path, 0)
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Function to rotate an image by a specified angle
    def rotate_image(image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image


        
    best_score = -1
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)
    best_rotation = 0
    
    for angle in [0, 90, 180, 270]:
        rotated_template = rotate_image(template, angle)
        
        # Perform template matching
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_top_left = max_loc
            best_bottom_right = (best_top_left[0] + rotated_template.shape[1], best_top_left[1] + rotated_template.shape[0])
            best_rotation = angle
    
    
    if best_score > 0.5:
        return True
    else:
        return False
