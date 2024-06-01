import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def create_heatmap(points, width, height):
    heatmap = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        heatmap[y, x] += 1
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    return heatmap

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    eye_positions = []
    for (ex, ey, ew, eh) in eyes:
        eye_positions.append((ex + ew // 2, ey + eh // 2))
    return eye_positions

def main():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    seconds_to_run = 40
    num_frames = fps * seconds_to_run
    eye_positions_all_frames = []

    heatmap_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        eye_positions = detect_eyes(frame)
        eye_positions_all_frames.extend(eye_positions)

        if len(eye_positions_all_frames) > 1:
            for i in range(1, len(eye_positions_all_frames)):
                cv2.line(frame, eye_positions_all_frames[i - 1], eye_positions_all_frames[i], (0, 255, 0), 2)

        for pos in eye_positions:
            cv2.circle(heatmap_overlay, pos, 2, (0, 0, 255), -1)

        heatmap_with_transparency = cv2.addWeighted(frame, 0.7, heatmap_overlay, 0.3, 0)
        cv2.imshow('frame', heatmap_with_transparency)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Create heatmap
    heatmap = create_heatmap(eye_positions_all_frames, width, height)

    # Save heatmap
    cv2.imwrite('heatmap.jpg', heatmap)

if __name__ == "__main__":
    main()
