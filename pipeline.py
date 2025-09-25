import cv2
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR
import json
import os
import numpy as np

# ========== Step 1. 提取线框 ==========
def extract_grid(img_path):
    img = cv2.imread(img_path, 0)  # 灰度
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 水平线
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # 垂直线
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # 提取坐标
    def extract_lines(binary_img, axis=0):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if axis == 0:  # vertical
                coords.append(x)
            else:          # horizontal
                coords.append(y)
        return sorted(list(set(coords)))

    xs = extract_lines(v_lines, axis=0)
    ys = extract_lines(h_lines, axis=1)

    # 构造网格
    cells = [[(xs[j], ys[i], xs[j+1], ys[i+1])
              for j in range(len(xs)-1)]
              for i in range(len(ys)-1)]

    return img, xs, ys, cells


# ========== Step 2. YOLO 检测单元格 ==========
def run_yolo(model_path, img_path, conf=0.25):
    model = YOLO(model_path)
    results = model.predict(source=img_path, conf=conf, verbose=False)

    yolo_boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():  # (x1,y1,x2,y2)
            x1, y1, x2, y2 = map(int, box[:4])
            yolo_boxes.append((x1, y1, x2, y2))
    return yolo_boxes


# ========== Step 3. 融合 YOLO 框和网格 ==========
def align_yolo_to_grid(yolo_boxes, xs, ys, cells):
    table = [[None for _ in range(len(xs)-1)] for _ in range(len(ys)-1)]

    for (x1, y1, x2, y2) in yolo_boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for i in range(len(ys)-1):
            for j in range(len(xs)-1):
                gx1, gy1, gx2, gy2 = cells[i][j]
                if gx1 <= cx < gx2 and gy1 <= cy < gy2:
                    table[i][j] = (x1, y1, x2, y2)
                    break
    return table


# ========== Step 4. OCR + JSON覆盖保存 ==========
def save_ocr_result_to_json(result, json_path):
    """保存 PaddleOCR predict() 的结果到 JSON（覆盖模式），自动转换 numpy 类型"""
    def default_converter(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result[0], f, ensure_ascii=False, indent=2, default=default_converter)



def extract_text_from_json(json_path):
    """从 OCR JSON 中提取文字"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "rec_texts" in data:
        return " ".join(data["rec_texts"])
    return ""


def table_to_dataframe(img_path, table, ocr_engine, json_path="ocr_result.json"):
    img = cv2.imread(img_path)
    rows = []
    for i, row in enumerate(table):
        row_values = []
        for j, cell in enumerate(row):
            if cell is None:
                row_values.append("")
            else:
                x1, y1, x2, y2 = cell
                crop = img[y1:y2, x1:x2]

                # OCR 识别
                result = ocr_engine.predict(crop)

                # 每次覆盖保存 JSON
                save_ocr_result_to_json(result, json_path)

                # 提取文字
                text = extract_text_from_json(json_path)
                row_values.append(text)

        rows.append(row_values)

    df = pd.DataFrame(rows)
    return df


# ========== Step 5. 可视化 ==========
def visualize(img, table, save_path="aligned_result.jpg"):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            if cell is not None:
                x1, y1, x2, y2 = cell
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{i},{j}", (x1+2, y1+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(save_path, vis)


# ========== 主程序 ==========
if __name__ == "__main__":
    img_path = "/home/zyb/百度飞桨领航团/学习项目/CV项目/table_ana/table_gen_dataset/table_img/610.jpg"
    model_path = "/home/zyb/百度飞桨领航团/学习项目/CV项目/table_ana/yolov12/runs/detect/train/weights/best.pt"

    # 1. 提取表格网格
    img, xs, ys, cells = extract_grid(img_path)

    # 2. YOLO 检测单元格
    yolo_boxes = run_yolo(model_path, img_path, conf=0.25)

    # 3. 对齐 YOLO 框到网格
    table = align_yolo_to_grid(yolo_boxes, xs, ys, cells)

    # 4. OCR + JSON覆盖
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang="ch")
    df = table_to_dataframe(img_path, table, ocr_engine, json_path="ocr_result.json")
    print(df)
    df.to_excel("output.xlsx", index=False, header=False)

    # 5. 可视化
    visualize(img, table)
    print("✅ 已保存 Excel，OCR 结果（ocr_result.json 每次覆盖），以及可视化结果")

