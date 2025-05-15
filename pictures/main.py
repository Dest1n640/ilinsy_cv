import cv2
import numpy as np

def is_car_like(contours, hierarchy):
    round_contours = []  # Для колёс и головы
    rect_contours = []   # Для кузова и кабины
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        circularity = (4 * np.pi * area) / (peri**2 + 1e-5)
        
        if 0.5 < circularity < 1.4 and area < 1000:
            # Круглые объекты (колёса и голова)
            round_contours.append((cnt, area, circularity))
        elif len(approx) >= 4 and len(approx) <= 10 and area > 1000:
            # Прямоугольные объекты (кузов, кабина)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            rect_contours.append((cnt, area, aspect_ratio, (x, y, w, h)))
    
    if len(round_contours) < 2 or len(rect_contours) < 1:
        return False
    
    rect_contours.sort(key=lambda x: x[1], reverse=True)
    
    round_contours.sort(key=lambda x: x[1])
    
    wheels = []
    for cnt, area, circ in round_contours:
        if 0.7 < circ < 1.3 and 50 < area < 800:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                wheels.append((cx, cy))
    
    if len(wheels) < 2:
        return False
    
    wheels_y = [w[1] for w in wheels]
    if max(wheels_y) - min(wheels_y) > 50:
        return False
    
    if len(rect_contours) == 0:
        return False
    
    _, _, body_ratio, (bx, by, bw, bh) = rect_contours[0]
    
    if not (1.5 < body_ratio < 4.0):
        return False
    
    wheels_under_body = 0
    for wx, wy in wheels:
        if bx <= wx <= bx + bw:
            if by + 0.5*bh <= wy <= by + 1.5*bh:
                wheels_under_body += 1
    
    if wheels_under_body < 2:
        return False
    
    if len(wheels) > 6:
        return False
    
    return True

video_path = "./output-1.avi"
cap = cv2.VideoCapture(video_path)
matched_frames = 0
total_frames = 0
prev_matched = False  # Для отслеживания последовательных совпадений
consecutive_matches = 0  # Для подсчета серий кадров

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    total_frames += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if len(frame.shape) == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        if np.mean(saturation) > 30:
            prev_matched = False
            consecutive_matches = 0
            continue
    
    methods_detected = 0
    
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    for thresh in [thresh1, thresh2]:
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if is_car_like(contours, hierarchy):
            methods_detected += 1
    
    if methods_detected > 0:
        if not prev_matched:
            matched_frames += 1
            prev_matched = True
            consecutive_matches = 1
        else:
            consecutive_matches += 1
            if consecutive_matches % 3 == 0:
                matched_frames += 1
    else:
        prev_matched = False
        consecutive_matches = 0

cap.release()
print(f"Всего кадров: {total_frames}")
print(f"Кадров с моим изображением: {matched_frames}")

