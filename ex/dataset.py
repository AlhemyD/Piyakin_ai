import os
import numpy as np
from skimage import draw
from scipy.ndimage import rotate
from PIL import Image
import matplotlib.pyplot as plt

base_dir = "dataset"
os.makedirs(base_dir, exist_ok=True)
for shape in ["square", "triangle", "circle"]:
    os.makedirs(f"{base_dir}/{shape}/images", exist_ok=True)
    os.makedirs(f"{base_dir}/{shape}/labels", exist_ok=True)

IMG_SIZE = 128
NUM_SAMPLES = 1000

class_map = {
    "square": 0,
    "triangle": 1,
    "circle": 2
}

def generate_figure(img_size, size, center_x, center_y, figure=""):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    bbox = None
    if figure == "square":
        half_size = size / 2
        start_y = max(0, int(center_y - half_size))#Начальная точка Y (не меньше 0)
        end_y = min(img_size - 1, int(center_y + half_size))#Конечная точка Y (не больше IMG_SIZE)
        
        start_x = max(0, int(center_x - half_size))#Начальная точка X (не меньше 0)
        end_x = min(img_size - 1, int(center_x + half_size))#Конечная точка X (не больше IMG_SIZE)
        
        rr, cc = draw.rectangle_perimeter(start=(start_y, start_x),
                                end=(end_y, end_x),
                                shape=img.shape)
        bbox = (start_x, start_y, end_x, end_y)
        img[rr, cc] = 255
    elif figure == "triangle":
        vertices = np.array([
            [center_y - size / 2, center_x],
            [center_y + size / 2, center_x - size / 2],
            [center_y + size / 2, center_x + size / 2]
        ])
        # Ограничиваем координаты, чтобы не выходить за пределы изображения
        vertices[:, 0] = np.clip(vertices[:, 0], 0, img_size - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, img_size - 1)
        
        # Рисуем только линии между вершинами для создания контура треугольника
        rr, cc = draw.line(vertices[0, 0].astype(int), vertices[0, 1].astype(int),
                           vertices[1, 0].astype(int), vertices[1, 1].astype(int))
        img[rr, cc] = 255
        rr, cc = draw.line(vertices[1, 0].astype(int), vertices[1, 1].astype(int),
                           vertices[2, 0].astype(int), vertices[2, 1].astype(int))
        img[rr, cc] = 255
        rr, cc = draw.line(vertices[2, 0].astype(int), vertices[2, 1].astype(int),
                           vertices[0, 0].astype(int), vertices[0, 1].astype(int))
        
        img[rr, cc] = 255
        start_x = min(vertices[:, 1])
        end_x = max(vertices[:, 1])
        start_y = min(vertices[:, 0])
        end_y = max(vertices[:, 0])
        bbox = (start_x, start_y, end_x, end_y)
        
    elif figure == "circle":
        radius=size//2
        rr,cc=draw.circle_perimeter(int(center_y), int(center_x), radius, shape=img.shape)
        img[rr, cc] = 255
        start_x = center_x - radius
        start_y = center_y - radius
        end_x = center_x + radius
        end_y = center_y + radius
        bbox = (start_x, start_y, end_x, end_y)
    return img, bbox

def save_yolo_label(bbox, class_id, img_size, filepath):
    start_x, start_y, end_x, end_y = bbox
    width = end_x - start_x
    height = end_y - start_y
    x_center = (start_x + width / 2) / img_size
    y_center = (start_y + height / 2) / img_size
    width_norm = width / img_size
    height_norm = height / img_size
    with open(filepath, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")


for shape in ["square", "triangle", "circle"]:
    for i in range(NUM_SAMPLES):
        size = np.random.randint(20, IMG_SIZE // 2)
        center_x = np.random.randint(size, IMG_SIZE - size)
        center_y = np.random.randint(size, IMG_SIZE - size)
        
        img, bbox = generate_figure(IMG_SIZE, size, center_x, center_y, shape)

        Image.fromarray(img).save(f"{base_dir}/{shape}/images/{i}.png")
        save_yolo_label(bbox, class_map[shape], IMG_SIZE, f"{base_dir}/{shape}/labels/{i}.txt")

print("Датасет и разметка успешно сгенерирован!")
