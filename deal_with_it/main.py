import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)


def censor(image, size=(10, 10)):
    result = np.zeros_like(image)
    stepy = result.shape[0] // size[0]
    stepx = result.shape[1] // size[1]
    for y in range(0, image.shape[0], stepy):
        for x in range(0, image.shape[1], stepx):
            result[y : y + stepy, x : x + stepx] = np.mean(
                image[y : y + stepy, x : x + stepx]
            )
    return result


def overlay_transparent(bg, fg, x, y):
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = fg.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fg_w, bg_w), min(y + fg_h, bg_h)
    if x1 >= x2 or y1 >= y2:
        return bg

    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    fg_rgb = fg[fy1:fy2, fx1:fx2, :3].astype(float)
    alpha = fg[fy1:fy2, fx1:fx2, 3].astype(float) / 255.0
    alpha = alpha[:, :, np.newaxis]

    bg_roi = bg[y1:y2, x1:x2].astype(float)

    blended = alpha * fg_rgb + (1 - alpha) * bg_roi
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 500)

face_cascade = cv2.CascadeClassifier("./xml/haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("./xml/haarcascade-eye.xml")

glasses = cv2.imread("./xml/deal-with-it.png", cv2.IMREAD_UNCHANGED)
h_g, w_g = glasses.shape[:2]

while capture.isOpened():
    ret, frame = capture.read()
    key = chr(cv2.waitKey(1) & 0xFF)
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes

        c1 = (x1 + w1 // 2, y1 + h1 // 2)
        c2 = (x2 + w2 // 2, y2 + h2 // 2)

        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        dist = np.hypot(dx, dy)
        scale = (2.0 * dist) / w_g

        # Поворачиваем и масштабируем очки
        center = (w_g // 2, h_g // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        glasses_tf = cv2.warpAffine(
            glasses,
            M,
            (w_g, h_g),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        h_t, w_t = glasses_tf.shape[:2]
        x = int((c1[0] + c2[0]) / 2 - w_t / 2)
        y = int((c1[1] + c2[1]) / 2 - h_t / 2)

        overlay_transparent(frame, glasses_tf, x, y)
    if key == "q":
        break
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()
