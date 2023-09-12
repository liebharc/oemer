from oemer import layers
import cv2
import numpy as np

def calculate_region_of_interest(detected_layers: list[np.ndarray]) -> np.ndarray:
    staff_and_notehead = detected_layers[0].astype(np.float32)
    for layer in detected_layers[1:]:
        staff_and_notehead += layer.astype(np.float32)
    dilation_shape = cv2.MORPH_ELLIPSE
    dilatation_size = 51
    dilatation_element = cv2.getStructuringElement(dilation_shape, (1, 2 * dilatation_size + 1))
    staff_and_notehead_diliated = cv2.dilate(staff_and_notehead, dilatation_element)

    erosion_size = dilatation_size
    erosion_element = cv2.getStructuringElement(dilation_shape, (15, 2 * erosion_size + 1))
    staff_and_notehead_eroded = cv2.erode(staff_and_notehead_diliated, erosion_element)

    dilatation_size = erosion_size + 5
    dilatation_element = cv2.getStructuringElement(dilation_shape, (20, 4 * dilatation_size + 1))
    
    region_of_interest = cv2.dilate(staff_and_notehead_eroded, dilatation_element)
    region_of_interest = cv2.threshold(region_of_interest, 0.1, 1, cv2.THRESH_BINARY)[1]
    return region_of_interest.astype(np.uint8)