import mediapipe as mp
import numpy as np
import cv2
import math, time

frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 2
FONTS = cv2.FONT_HERSHEY_COMPLEX
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
sketch_face = mp_draw.DrawingSpec(thickness=1, circle_radius=0, color=(0, 255, 0))
coordinates = mp_draw._normalized_to_pixel_coordinates

cap = cv2.VideoCapture(0)

start_time = time.time()

# Smoothing parameters
ALPHA = 0.1
blink_ratio_smoothed = 0

def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    return math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

while cap.isOpened():
    frame_counter += 1
    _, image = cap.read()
    image = np.ascontiguousarray(image)
    frame_height, frame_width = image.shape[:2]

    result = mp_face_mesh.FaceMesh(refine_landmarks=True).process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        mesh_coord = [(int(point.x * frame_width), int(point.y * frame_height)) for point in face_landmarks.landmark]

        # Calculate blink ratio using Numpy
        rh_right = np.array(mesh_coord[RIGHT_EYE[0]])
        rh_left = np.array(mesh_coord[RIGHT_EYE[8]])
        rv_top = np.array(mesh_coord[RIGHT_EYE[12]])
        rv_bottom = np.array(mesh_coord[RIGHT_EYE[4]])

        lh_right = np.array(mesh_coord[LEFT_EYE[0]])
        lh_left = np.array(mesh_coord[LEFT_EYE[8]])
        lv_top = np.array(mesh_coord[LEFT_EYE[12]])
        lv_bottom = np.array(mesh_coord[LEFT_EYE[4]])

        rh_distance = euclidean_distance(rh_right, rh_left)
        rv_distance = euclidean_distance(rv_top, rv_bottom)
        lv_distance = euclidean_distance(lv_top, lv_bottom)
        lh_distance = euclidean_distance(lh_right, lh_left)

        # Check for zero distance to avoid division by zero
        if rv_distance != 0:
            re_ratio = rh_distance / rv_distance
        else:
            re_ratio = 0

        if lv_distance != 0:
            le_ratio = lh_distance / lv_distance
        else:
            le_ratio = 0

        ratio = (re_ratio + le_ratio) / 2

        # Smooth the blink ratio
        blink_ratio_smoothed = (1 - ALPHA) * blink_ratio_smoothed + ALPHA * ratio

        cv2.putText(image, f'Ratio: {round(blink_ratio_smoothed, 2)}', (30, 100), FONTS, 0.7, (0, 255, 255), 2)

        if blink_ratio_smoothed > 5.5:
            CEF_COUNTER += 1
            cv2.putText(image, 'Blink', (int(frame_width / 2), 100), FONTS, 1.7, (0, 255, 255), 2)
        else:
            if CEF_COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                CEF_COUNTER = 0

        cv2.putText(image, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONTS, 0.7, (0, 255, 0), 2)

        cv2.polylines(image, [np.array([mesh_coord[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1,
                      cv2.LINE_AA)
        cv2.polylines(image, [np.array([mesh_coord[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1,
                      cv2.LINE_AA)

        mp_draw.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_LEFT_EYE, None, sketch_face)
        mp_draw.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_RIGHT_EYE, None, sketch_face)
        mp_draw.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_LIPS, None, sketch_face)

    cv2.imshow("Testing the core", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
