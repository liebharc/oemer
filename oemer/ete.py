import os
import pickle
import argparse
import urllib.request
from pathlib import Path
from typing import Tuple
from argparse import Namespace, ArgumentParser

from PIL import Image
from numpy import ndarray

from oemer.region_of_interest import calculate_region_of_interest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from oemer import MODULE_PATH
from oemer import layers
from oemer.inference import inference
from oemer.logger import DEBUG_LEVEL, DEBUG_OUTPUT, WINDOW_NAME, debug_show, get_logger, set_debug_level
from oemer.dewarp import dewarp_rgb, estimate_coords, dewarp
from oemer.staffline_extraction import extract as staff_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.note_group_extraction import extract as group_extract
from oemer.symbol_extraction import extract as symbol_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.build_system import MusicXMLBuilder
from oemer.draw_teaser import draw_notes, draw_before_rhythm, draw_staff_and_zones, teaser


logger = get_logger(__name__)


CHECKPOINTS_URL = {
    "1st_model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_model.onnx",
    "1st_weights.h5": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_weights.h5",
    "2nd_model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx",
    "2nd_weights.h5": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_weights.h5"
}



def clear_data() -> None:
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)

def generate_pred_staff(image: np.ndarray, use_tf: bool = False) -> Tuple[ndarray, ndarray]:
    logger.info("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/unet_big"),
        image,
        use_tf=use_tf,
    )
    staff = np.where(staff_symbols_map==1, 1, 0)
    symbols = np.where(staff_symbols_map==2, 1, 0)
    return staff, symbols


def generate_pred(image: np.ndarray, use_tf: bool = False) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    staff, symbols = generate_pred_staff(image, use_tf=use_tf)
    logger.info("Extracting layers of different symbols")
    sep, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/seg_net"),
        image,
        manual_th=None,
        use_tf=use_tf,
    )
    stems_rests = np.where(sep==1, 1, 0)
    notehead = np.where(sep==2, 1, 0)
    clefs_keys = np.where(sep==3, 1, 0)
    # stems_rests = sep[..., 0]
    # notehead = sep[..., 1]
    # clefs_keys = sep[..., 2]

    return staff, symbols, stems_rests, notehead, clefs_keys


def polish_symbols(rgb_black_th=300):
    img = layers.get_layer('original_image')
    sym_pred = layers.get_layer('symbols_pred')

    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0]))
    arr = np.sum(np.array(img), axis=-1)
    arr = np.where(arr < rgb_black_th, 1, 0)  # Filter background
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker)  # Filter staff lines
    mix = np.where(sym_pred+arr>1, 1, 0)
    return mix


def register_notehead_bbox(bboxes):
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('bboxes')
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2])
    return layer


def register_note_id() -> None:
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def extract(args: Namespace) -> str:
    img_path = Path(args.img_path)
    f_name = os.path.splitext(img_path.name)[0]
    pkl_path = img_path.parent / f"{f_name}.pkl"
    original_image = cv2.imread(str(img_path))
    if pkl_path.exists():
        # Load from cache
        logger.info("Loading from cache")
        pred = pickle.load(open(pkl_path, "rb"))
        notehead = pred["note"]
        symbols = pred["symbols"]
        staff = pred["staff"]
        clefs_keys = pred["clefs_keys"]
        stems_rests = pred["stems_rests"]
    else:
        # Make predictions
        if args.use_tf:
            ori_inf_type = os.environ.get("INFERENCE_WITH_TF", None)
            os.environ["INFERENCE_WITH_TF"] = "true"
        if args.two_pass_deskew:
            # Pass 1
            logger.info("Staffline Pass 1")
            staff, symbols = generate_pred_staff(original_image, use_tf=args.use_tf)
            # Pass 2
            logger.info("Staffline Pass 2")
            coords_x, coords_y = estimate_coords(staff)
            original_image = dewarp_rgb(original_image, coords_x, coords_y)
            debug_show(f_name, 1.0, 'input_dewarp_pass1', original_image)
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(original_image, use_tf=args.use_tf)
        if args.use_tf and ori_inf_type is not None:
            os.environ["INFERENCE_WITH_TF"] = ori_inf_type
        if args.save_cache:
            data = {
                'staff': staff,
                'note': notehead,
                'symbols': symbols,
                'stems_rests': stems_rests,
                'clefs_keys': clefs_keys
            }
            pickle.dump(data, open(pkl_path, "wb"))

    # Load the original image, resize to the same size as prediction.
    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))
    debug_show(f_name, 0.0, 'original', original_image)
    debug_show(f_name, 0.0, 'staff', staff, scale=True)
    debug_show(f_name, 0.0, 'notehead', notehead, scale=True)
    debug_show(f_name, 0.0, 'symbols', symbols, scale=True)
    debug_show(f_name, 0.0, 'stems_rests', stems_rests, scale=True)
    debug_show(f_name, 0.0, 'clefs_keys', clefs_keys, scale=True)

    if not args.without_deskew:
        try:
            logger.info("Dewarping")
            coords_x, coords_y = estimate_coords(staff)
            staff_dewarped = dewarp(staff, coords_x, coords_y)
            symbols_dewarped = dewarp(symbols, coords_x, coords_y)
            stems_rests_dewarped = dewarp(stems_rests, coords_x, coords_y)
            clefs_keys_dewarped = dewarp(clefs_keys, coords_x, coords_y)
            notehead_dewarped = dewarp(notehead, coords_x, coords_y)
            image_dewarped = original_image.copy()
            for i in range(original_image.shape[2]):
                image_dewarped[..., i] = dewarp(original_image[..., i], coords_x, coords_y)
            logger.info("Dewarping done")
            staff = staff_dewarped
            symbols = symbols_dewarped
            stems_rests = stems_rests_dewarped
            clefs_keys = clefs_keys_dewarped
            notehead = notehead_dewarped
            original_image = image_dewarped
        except Exception as e:
            logger.error("Dewarping failed, skipping.")

    # Register predictions
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols>1] = 1
    roi = calculate_region_of_interest([notehead, staff, clefs_keys])
    stems_rests = cv2.bitwise_and(stems_rests, stems_rests, mask = roi)
    clefs_keys = cv2.bitwise_and(clefs_keys, clefs_keys, mask = roi)
    notehead = cv2.bitwise_and(notehead, notehead, mask = roi)
    symbols = cv2.bitwise_and(symbols, symbols, mask = roi)
    staff = cv2.bitwise_and(staff, staff, mask = roi)
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", original_image)
    
    debug_show(f_name, 1.0, 'original', original_image)
    debug_show(f_name, 1.0, 'staff', staff, scale=True)
    debug_show(f_name, 1.0, 'notehead', notehead, scale=True)
    debug_show(f_name, 1.0, 'symbols', symbols, scale=True)
    debug_show(f_name, 1.0, 'stems_rests', stems_rests, scale=True)
    debug_show(f_name, 1.0, 'clefs_keys', clefs_keys, scale=True)
    debug_show(f_name, 1.0, 'roi', roi, scale=True)

    # ---- Extract staff lines and group informations ---- #
    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object.
    debug_show(f_name, 2.0, 'staffs', draw_staff_and_zones())

    # ---- Extract noteheads ---- #
    logger.info("Extracting noteheads")
    notes = note_extract()

    # Array of 'NoteHead' instances.
    layers.register_layer('notes', np.array(notes))
    debug_show(f_name, 2.0, 'notes', draw_notes())

    # Add a new layer (w * h), indicating note id of each pixel.
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64)-1)
    register_note_id()

    # ---- Extract groups of note ---- #
    logger.info("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    # ---- Extract symbols ---- #
    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))
    debug_show(f_name, 2.0, 'before_rhythm', draw_before_rhythm())

    # ---- Parse rhythm ---- #
    logger.info("Extracting rhythm types")
    rhythm_extract()

    # ---- Build MusicXML ---- #
    logger.info("Building MusicXML document")
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder = MusicXMLBuilder(title=basename.capitalize(), assume_simple=args.assume_simple)
    builder.build()
    xml = builder.to_musicxml()

    # ---- Write out the MusicXML ---- #
    out_path = args.output_path
    if not out_path.endswith(".musicxml"):
        # Take the output path as the folder.
        out_path = os.path.join(out_path, basename+".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)

    return out_path


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        "Oemer",
        description="End-to-end OMR command line tool. Receives an image as input, and outputs MusicXML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("img_path", help="Path to the image.", type=str, nargs='?')
    parser.add_argument(
        "-o", "--output-path", help="Path to output the result file.", type=str, default="./")
    parser.add_argument(
        "--use-tf", help="Use Tensorflow for model inference. Default is to use Onnxruntime.", action="store_true")
    parser.add_argument(
        "--save-cache",
        help="Save the model predictions and the next time won't need to predict again.",
        action='store_true')
    parser.add_argument(
        "-d",
        "--without-deskew",
        help="Disable the deskewing step if you are sure the image has no skew.",
        action='store_true')
    parser.add_argument(
        "--two-pass-deskew",
        help="Deskew in two passes, might improve deskew results.",
        action='store_true')
    parser.add_argument(
        "--debug",
        help="Enable debug mode. The debug images will be saved to the current directory.",
        action='store_true'
    )
    parser.add_argument(
        "--assume-simple",
        help="Instruct the detection to assume simple sheet music. This increases the robustness. Assumptions are:\n- only one key",
        action='store_true'
    )
    parser.add_argument(
        "--just-download-model",
        help="Just downloads the models and exits afterwards. Other arguments will be ignored.",
        action='store_true')
    return parser


def download_file(title: str, url: str, save_path: str) -> None:
    resp = urllib.request.urlopen(url)
    length = int(resp.getheader("Content-Length", -1))

    chunk_size = 2**9
    total = 0
    with open(save_path, "wb") as out:
        while True:
            print(f"{title}: {total*100/length:.1f}% {total}/{length}", end="\r")
            data = resp.read(chunk_size)
            if not data:
                break
            total += out.write(data)
        print(f"{title}: 100% {length}/{length}"+" "*20)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    if args.debug:
        set_debug_level(1)

    if not args.just_download_model:
        if not args.img_path:
            raise ValueError("Please specify the image path.")
        if not os.path.exists(args.img_path):
            raise FileNotFoundError(f"The given image path doesn't exists: {args.img_path}")

    # Check there are checkpoints
    chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/weights.h5")
    if not os.path.exists(chk_path):
        logger.warn("No checkpoint found in %s", chk_path)
        for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
            if ".onnx" in title and args.use_tf:
                continue
            logger.info(f"Downloading checkpoints ({idx+1}/{len(CHECKPOINTS_URL)})")
            save_dir = "unet_big" if title.startswith("1st") else "seg_net"
            save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
            save_path = os.path.join(save_dir, title.split("_")[1])
            download_file(title, url, save_path)

    if args.just_download_model:
        logger.info("Downloaded model, exiting now as --just-download-model is set.")
        return

    clear_data()
    mxl_path = extract(args)
    img = teaser()
    img.save(mxl_path.replace(".musicxml", "_teaser.png"))


if __name__ == "__main__":
    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)
    main()
