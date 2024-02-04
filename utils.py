import pickle

from skimage.transform import resize
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

model = pickle.load(open("model.p", "rb"))

def is_empty(spot_bgr):
    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    model_output = model.predict(flat_data)

    if model_output:
        return EMPTY
    
    return NOT_EMPTY

def get_parking_spots_bboxes(connected_components):
    totalLabels, label_ids, values, centroid = connected_components

    slots = []
    coef = 1

    for i in range(1, totalLabels):
        x = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y = int(values[i, cv2.CC_STAT_TOP] * coef)
        width = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        height = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x, y, width, height])

    return slots
    