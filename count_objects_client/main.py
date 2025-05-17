import cv2
import zmq
import numpy as np

address = "84.237.21.36"
port = 6002

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect(f"tcp://{address}:{port}")

cv2.namedWindow("Client", cv2.WINDOW_GUI_NORMAL)
count = 0

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

while True:
    message = socket.recv()
    frame = cv2.imdecode(np.frombuffer(message, np.uint8), -1)
    count += 1

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_count = len(contours)
    circle = 0
    rectangle = 0

    for contour in contours:
        eps = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        for p in approx:
            cv2.circle(thresh, tuple(p[0]), 6, (0, 255, 0), 2)

        if len(approx) <= 5:
            rectangle += 1
        else:
            circle += 1

    cv2.putText(
        frame,
        f"Frame count: {count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Circles: {circle}",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Rectangles: {rectangle}",
        (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    cv2.imshow("Client", frame)
    cv2.imshow("Contours", thresh)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
