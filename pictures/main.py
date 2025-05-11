import cv2
import numpy as np


def is_car_like(contours, hierarchy):
    wheel_count = 0
    rect_count = 0
    head_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Поиск колёс — почти круги
        if len(approx) > 6 and area < 1000:
            wheel_count += 1

        # Поиск корпуса — прямоугольник
        elif len(approx) == 4 and area > 1000:
            rect_count += 1

        # Поиск головы — круглая маленькая фигура
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2 + 1e-5)
        if 0.7 < circularity < 1.2 and 50 < area < 400:
            head_detected = True

    return wheel_count >= 2 and rect_count >= 1 and head_detected


video_path = "./output-1.avi"
cap = cv2.VideoCapture(video_path)

matched_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if is_car_like(contours, hierarchy):
        matched_frames += 1

cap.release()
print("Кадров с моим изображением", matched_frames)
