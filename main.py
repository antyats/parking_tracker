import cv2 as cv

from utils import get_parking_spots_bboxes
from utils import is_empty

VIDEO_PATH = "./data/parking_crop_loop.mp4"
MASK_CROP_PATH = "./mask_crop.png"

mask = cv.imread(MASK_CROP_PATH, 0)
cap = cv.VideoCapture(VIDEO_PATH)

connected_components = cv.connectedComponentsWithStats(mask, 4, cv.CV_32S)

spots = get_parking_spots_bboxes(connected_components)
print(spots[0])

ret = True

while ret:
    ret, frame = cap.read()

    for spot in spots:
        x, y, w, h = spot

        spot_crop = frame[y:y + h, x:x + w, :]
        spot_status = is_empty(spot_crop)

        if spot_status:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),)

    cv.imshow("frame", frame)
    if cv.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()