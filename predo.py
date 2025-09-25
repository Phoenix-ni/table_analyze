import cv2
import numpy as np

img = cv2.imread("/home/zyb/百度飞桨领航团/学习项目/CV项目/table_analyze/table_gen_dataset/table_img/607.jpg", 0)  # 灰度
# 反色二值化，线条变成白色，背景黑
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ===== 提取水平方向线条 =====
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

# ===== 提取垂直方向线条 =====
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

# ===== 合并线条 =====
table_lines = cv2.add(h_lines, v_lines)

# ===== 可选：细化/加粗线条 =====
table_lines = cv2.dilate(table_lines, np.ones((2,2), np.uint8), iterations=1)

# ===== 创建一张纯白背景图，把线画上去 =====
clean_table = 255 * np.ones_like(img)  # 全白背景
clean_table[table_lines > 0] = 0       # 线条部分设为黑色

# 保存结果
cv2.imwrite("clean_table.jpg", clean_table)

# 显示结果
cv2.imshow("Clean Table", clean_table)
cv2.waitKey(0)
cv2.destroyAllWindows()
