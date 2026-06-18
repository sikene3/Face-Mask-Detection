import os
import random
import shutil
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "raw", "annotations")
YOLO_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

TRAIN_SPLIT = 0.8
random.seed(42)


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    yolo_lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_id = CLASS_TO_ID[name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        x_center = round(x_center, 6)
        y_center = round(y_center, 6)
        width = round(width, 6)
        height = round(height, 6)
        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return yolo_lines


def main():
    for subdir in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        os.makedirs(os.path.join(YOLO_DIR, subdir), exist_ok=True)

    xml_files = sorted(
        [f for f in os.listdir(ANNOTATIONS_DIR) if f.lower().endswith(".xml")]
    )
    random.shuffle(xml_files)

    split_idx = int(len(xml_files) * TRAIN_SPLIT)
    train_xmls = xml_files[:split_idx]
    val_xmls = xml_files[split_idx:]

    for split_name, xml_list in [("train", train_xmls), ("val", val_xmls)]:
        img_dst_dir = os.path.join(YOLO_DIR, "images", split_name)
        lbl_dst_dir = os.path.join(YOLO_DIR, "labels", split_name)
        for xml_file in xml_list:
            base_name = os.path.splitext(xml_file)[0]
            xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)

            yolo_lines = parse_xml(xml_path)

            for ext in [".png", ".jpg", ".jpeg"]:
                src_img = os.path.join(IMAGES_DIR, base_name + ext)
                if os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(img_dst_dir, base_name + ext))
                    break

            lbl_path = os.path.join(lbl_dst_dir, base_name + ".txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

    yaml_content = (
        "train: ../data/yolo_dataset/images/train\n"
        "val: ../data/yolo_dataset/images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    yaml_path = os.path.join(BASE_DIR, "config", "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Done. {len(train_xmls)} train, {len(val_xmls)} val samples.")
    print(f"YOLO dataset at: {YOLO_DIR}")
    print(f"data.yaml at: {yaml_path}")


if __name__ == "__main__":
    main()
