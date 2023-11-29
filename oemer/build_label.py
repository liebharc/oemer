import os
import random
from PIL import Image

import cv2
import numpy as np

from .constant_min import CLASS_CHANNEL_MAP


HALF_WHOLE_NOTE = [39, 41, 42, 43, 45, 46, 47, 49]
STAFF_COLOR = 165


def fill_hole(gt, tar_color):
    assert tar_color in HALF_WHOLE_NOTE
    tar = np.where(gt==tar_color, 1, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(tar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        # Scan by row
        for yi in range(y, y+h):
            cur = x
            cand_y = []
            cand_x = []
            while cur <= x+w:
                if tar[yi, cur] > 0:
                    break
                cur += 1
            while cur <= x+w:
                if tar[yi, cur] == 0:
                    break
                cur += 1
            while cur <= x+w:
                if tar[yi, cur] > 0:
                    break
                cand_y.append(yi)
                cand_x.append(cur)
                cur += 1
            if cur <= x+w:
                cand_y = np.array(cand_y)
                cand_x = np.array(cand_x)
                tar[cand_y, cand_x] = 1

        # Scan by column
        for xi in range(x, x+w):
            cur = y
            cand_y = []
            cand_x = []
            while cur <= y+h:
                if tar[cur, xi] > 0:
                    break
                cur += 1
            while cur <= y+h:
                if tar[cur, xi] == 0:
                    break
                cur += 1
            while cur <= y+h:
                if tar[cur, xi] > 0:
                    break
                cand_y.append(cur)
                cand_x.append(xi)
                cur += 1
            if cur <= y+h:
                cand_y = np.array(cand_y)
                cand_x = np.array(cand_x)
                tar[cand_y, cand_x] = 1

    return tar


def build_label(seg_path):
    img = Image.open(seg_path)
    arr = np.array(img)
    color_set = set(np.unique(arr))
    color_set.remove(0)  # Remove background color from the candidates

    total_chs = len(set(CLASS_CHANNEL_MAP.values())) + 2  # Plus 'background' and 'others' channel.
    output = np.zeros(arr.shape + (total_chs,))

    output[..., 0] = np.where(arr==0, 1, 0)
    for color in color_set:
        ch = CLASS_CHANNEL_MAP.get(color, -1)
        #if (ch != 0) and color in HALF_WHOLE_NOTE:
        #    note = fill_hole(arr, color)
        #    output[..., ch] += note
        #else:
        if (ch != 0) and color == STAFF_COLOR:
            lines_closed = close_lines(arr)
            output[..., ch] += np.where(lines_closed==255, 1, 0)
        else:
            output[..., ch] += np.where(arr==color, 1, 0)
    return output


def find_example(dataset_path: str, color: int, max_count=500, mark_value=255):
    files = os.listdir(dataset_path)
    random.shuffle(files)
    for ff in files[:max_count]:
        path = os.path.join(dataset_path, ff)
        img = Image.open(path)
        arr = np.array(img)
        if color in arr:
            color_set = set(np.unique(arr))
            print(sorted(list(color_set)), color, len(color_set))
            return np.where(arr==color, mark_value, arr / 2), np.where(arr==color, mark_value, 0).astype(np.uint8)

def close_morph(img: np.ndarray):
    kernel = np.ones((3, 15), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def close_lines(img: np.ndarray):
    #img = close_morph(img)
    # Use hough transform to find lines
    width = img.shape[1]
    lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=width//32, minLineLength=width//16, maxLineGap=500)
    if lines is not None:
        angles = []
        # Draw lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1)
            angles.append(angle)
        mean_angle = np.mean(angles)
        # Draw lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1)
            is_horizontal = abs(angle - mean_angle) < np.pi/4
            if is_horizontal:
                cv2.line(img, (x1,y1), (x2,y2), 255, 1)
    else:
        print("No lines found")

    return img

if __name__ == "__main__":
    seg_folder = 'ds2_dense/segmentation'
    import sys
    color = int(sys.argv[1])
    with_background, without_background = find_example(seg_folder, color)  # type: ignore
    cv2.imwrite("example.png", with_background)
    if color == STAFF_COLOR:
        lines_closed = close_lines(without_background)
        cv2.imwrite("example_closed.png", lines_closed)



