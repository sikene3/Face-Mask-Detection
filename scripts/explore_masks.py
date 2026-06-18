import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "raw", "annotations")

CLASS_COLORS = {
    "with_mask": (0, 255, 0),
    "without_mask": (0, 0, 255),
    "mask_weared_incorrect": (0, 255, 255),
}


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(name)
    return boxes, labels


def main():
    image_files = [
        f
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print("No image files found in", IMAGES_DIR)
        return

    chosen = random.choice(image_files)
    image_path = os.path.join(IMAGES_DIR, chosen)
    base_name = os.path.splitext(chosen)[0]
    xml_path = os.path.join(ANNOTATIONS_DIR, base_name + ".xml")

    if not os.path.exists(xml_path):
        print("Annotation not found for", chosen)
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image:", image_path)
        return

    boxes, labels = parse_xml(xml_path)

    for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = ymin - 4 if ymin - th - 4 > 0 else ymin + th + 4
        cv2.rectangle(
            img,
            (xmin, text_y - th - 2),
            (xmin + tw + 2, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (xmin + 1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
