from PIL import Image

import cv2
import numpy as np

from oemer import layers
from numpy import ndarray
from typing import List, Optional, Tuple, Union
import typing

# Globals
out: ndarray

@typing.no_type_check
def draw_bbox(bboxes: Union[List[Tuple[int, int, int, int]], List[ndarray]], color: Tuple[int, int, int], text: Optional[str] = None, labels: Optional[List[str]] = None, text_y_pos: float = 1) -> None:
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        (x1, y1, x2, y2) = (int(np.round(x1)), int(np.round(y1)), int(np.round(x2)), int(np.round(y2)))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        y_pos = y1 + round((y2-y1)*text_y_pos)
        if text is not None:
            cv2.putText(out, text, (x2+2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        else:
            cv2.putText(out, labels[idx], (x2+2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

def draw_staff_and_zones() -> Image.Image:
    ori_img = layers.get_layer('original_image')
    staffs = layers.get_layer('staffs')
    zones = layers.get_layer('zones')

    global out
    out = np.copy(ori_img).astype(np.uint8)

    draw_bbox([[gg.x_left, gg.y_upper, gg.x_right, gg.y_lower] for gg in staffs.flatten()], color=(255, 192, 92), text="staffs")
    draw_bbox([[gg.start, 0, gg.stop, 10] for gg in zones], color=(194, 81, 167), text="zones")
    return out

def draw_notes() -> Image.Image:
    ori_img = layers.get_layer('original_image')
    staffs = layers.get_layer('staffs')
    notes = layers.get_layer('notes')
    roi = layers.get_layer('roi')

    global out
    out = np.copy(ori_img).astype(np.uint8)

    draw_bbox([[gg.x_left, gg.y_upper, gg.x_right, gg.y_lower] for gg in staffs.flatten()], color=(255, 192, 92), text="staffs")
    for note in notes:
            if note.label is not None:
                x1, y1, x2, y2 = note.bbox
                cv2.putText(out, note.label.name[0], (x2+2, y2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)

    return out

def draw_before_rhythm() -> Image.Image:
    ori_img = layers.get_layer('original_image')
    staffs = layers.get_layer('staffs')
    barlines = layers.get_layer('barlines')
    clefs = layers.get_layer('clefs')
    accidentals = layers.get_layer('accidentals')
    rests = layers.get_layer('rests')
    notes = layers.get_layer('notes')

    global out
    out = np.copy(ori_img).astype(np.uint8)
    draw_bbox([b.bbox for b in barlines], color=(63, 87, 181), text='barline', text_y_pos=0.5)
    draw_bbox([s.bbox for s in accidentals if s.note_id is None], color=(90, 0, 168), labels=[str(s.label.name) for s in accidentals if s.note_id is None])
    draw_bbox([c.bbox for c in clefs], color=(235, 64, 52), labels=[c.label.name for c in clefs])
    draw_bbox([r.bbox for r in rests], color=(12, 145, 0), labels=[r.label.name for r in rests])    
    draw_bbox([[gg.x_left, gg.y_upper, gg.x_right, gg.y_lower] for gg in staffs.flatten()], color=(255, 192, 92), text="staffs")
    for note in notes:
            if note.label is not None:
                x1, y1, x2, y2 = note.bbox
                cv2.putText(out, str(note.staff_line_pos) + "-" + note.label.name[0], (x2+2, y2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)

    return out

def teaser() -> Image.Image:
    ori_img = layers.get_layer('original_image')
    notes = layers.get_layer('notes')
    groups = layers.get_layer('note_groups')
    barlines = layers.get_layer('barlines')
    clefs = layers.get_layer('clefs')
    accidentals = layers.get_layer('accidentals')
    rests = layers.get_layer('rests')

    global out
    out = np.copy(ori_img).astype(np.uint8)

    draw_bbox([gg.bbox for gg in groups], color=(255, 192, 92), text="group")
    draw_bbox([n.bbox for n in notes if not n.invalid], color=(194, 81, 167), labels=[str(n.label)[0] for n in notes if not n.invalid])
    draw_bbox([b.bbox for b in barlines], color=(63, 87, 181), text='barline', text_y_pos=0.5)
    draw_bbox([s.bbox for s in accidentals if s.note_id is None], color=(90, 0, 168), labels=[str(s.label.name) for s in accidentals if s.note_id is None])
    draw_bbox([c.bbox for c in clefs], color=(235, 64, 52), labels=[c.label.name for c in clefs])
    draw_bbox([r.bbox for r in rests], color=(12, 145, 0), labels=[r.label.name for r in rests])

    for note in notes:
        if note.label is not None:
            x1, y1, x2, y2 = note.bbox
            cv2.putText(out, note.label.name[0], (x2+2, y2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)

    return Image.fromarray(out)
