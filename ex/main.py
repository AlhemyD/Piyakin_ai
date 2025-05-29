import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


def preprocess_image(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.reshape(1, target_size[0], target_size[1], 1) / 255.0
    return img

def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

class DrawingApp:
    def __init__(self, model_path="shape_detector.keras", window_size=(128, 128)):
        self.window_size = window_size
        self.canvas = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        self.drawing = False
        self.last_point = None
        self.model = load_model(model_path)
        self.window_name = "Draw Shapes"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_callback)

    def draw_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.last_point is not None:
                cv2.line(self.canvas, self.last_point, (x, y), (255, 255, 255), 5)
                self.last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None

    def predict_shape(self):
        if np.sum(self.canvas) == 0:  # Если холст пустой
            return None
        input_img = preprocess_image(self.  canvas)
        bbox = self.model.predict(input_img)[0]  # Предсказание bounding box
        return bbox

    def clear_canvas(self):
        self.canvas = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

    def run(self):
        while True:
            display = self.canvas.copy()
            key = cv2.waitKey(1) & 0xFF
            bbox = self.predict_shape()
            if bbox is not None:
                display = draw_bbox(display, bbox)
            if key == ord('q'):  # Выход по нажатию 'q'
                break
            elif key == ord('c'):  # Очистка холста по нажатию 'c'
                self.clear_canvas()
            cv2.imshow(self.window_name, display)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DrawingApp()
    app.run()
