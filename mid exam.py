import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\a9006\Desktop\yolo\yolov8\HW\LaneVideo.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = r"C:\Users\a9006\Desktop\yolo\yolov8\HW\LaneDetection.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)

    roi_vertices = np.array([[
        (width * 0.1, height),
        (width * 0.45, height * 0.75),
        (width * 0.55, height * 0.75),
        (width * 0.9, height)
    ]], np.int32)

    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=200)

    line_frame = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return line_frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)

    cv2.imshow("Lane Detection", processed_frame)

    out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"影片儲存完成：{output_path}")
