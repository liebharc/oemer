import random
import pickle
import shutil
from os import remove
from pathlib import Path
from PIL import Image

import augly.image as imaugs
import numpy as np
import cv2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

from oemer.bbox import get_bbox, merge_nearby_bbox, draw_bounding_boxes, rm_merge_overlap_bbox
from oemer.build_label import find_example


SVM_PARAM_GRID = {
    'degree': [2, 3, 4],
    'decision_function_shape': ['ovo', 'ovr'],
    'C':[0.1, 1, 10, 100],
    'gamma':[0.0001, 0.001, 0.1, 1],
    'kernel':['rbf', 'poly']
}
TARGET_WIDTH = 40
TARGET_HEIGHT = 70
DISTANCE = 10

DATASET_PATH = "./ds2_dense/segmentation"


def _collect(colors, out_path, samples=100):
    print("Collecting", out_path)
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    idx = 0
    for color in colors:
        cur_samples = 0
        while cur_samples < samples / len(colors):
            arr = find_example(DATASET_PATH, color)
            if arr is None:
                continue
            arr[arr!=200] = 0
            boxes = get_bbox(arr)
            if len(boxes) > 1:
                boxes = merge_nearby_bbox(boxes, DISTANCE)
            boxes = rm_merge_overlap_bbox(boxes)
            for box in boxes:
                if idx >= samples:
                    break
                print(f"{idx+1}/{samples}", end='\r')
                add_space = random.randint(0, 5)
                patch = arr[box[1]-add_space:box[3]+add_space, box[0]-add_space:box[2]+add_space]
                ratio = random.choice(np.arange(0.6, 1.3, 0.1))
                tar_w = int(ratio * patch.shape[1])
                tar_h = int(ratio * patch.shape[0])
                img = imaugs.resize(Image.fromarray(patch.astype(np.uint8)), width=tar_w, height=tar_h)

                seed = random.randint(0, 1000)
                np.float = float # Monkey patch to workaround removal of np.float
                aug_image = imaugs.perspective_transform(img, seed=seed, sigma=3)    # Blur
                
                rad = random.choice(np.arange(0.0001, 2.1, 0.5))
                #aug_image = imaugs.blur(aug_image, radius=rad)

                # Pixel shuffle, kind of adding noise
                factor = random.choice(np.arange(0.0001, 0.26, 0.05))
                #aug_image = imaugs.shuffle_pixels(aug_image, factor=factor)

                # Image quality
                qa = random.randint(98, 100)
                aug_image = imaugs.encoding_quality(aug_image, quality=qa)

                # Opacity
                level = random.randint(6, 10) / 10
                #aug_image = imaugs.opacity(aug_image, level=level)
                
                # Pixelize (pretty similar to blur?)
                rat = random.randint(3, 10) / 10
                #aug_image = imaugs.pixelization(aug_image, ratio=rat)

                # Pixelize (pretty similar to blur?)
                rat = random.randint(2, 4) / 10
                #aug_image = imaugs.pixelization(aug_image, ratio=rat)

                img =  cv2.cvtColor(np.array(aug_image), cv2.COLOR_RGB2GRAY)
                img = np.where(np.array(img)>0, 255, 0)
                Image.fromarray(img.astype(np.uint8)).save(out_path / f"{idx}.png")
                idx += 1

            cur_samples += len(boxes)
    print()


def collect_data(samples=400):
    # the color is the gray scale value of the pixels in the segmentation image
    color_map = {
        "sharp": [74],
        "flat": [70],
        "natural": [72],
        'rest_whole': [97],
        'rest_half': [98],
        'rest_quarter': [99],
        'rest_8th': [100],
        'rest_16th': [101],
        'rest_32nd': [102],
        'rest_64th': [103],
        'gclef': [10],
        'fclef': [13],
        "notehead_solid": [35, 37],
        "notehead_hollow": [39, 41],
    }

    for name, colors in color_map.items():
        print('Current', name)
        _collect(colors, f"train_data/{name}", samples=samples)
        _collect(colors, f"test_data/{name}", samples=samples)


def train(folders):
    class_map = {idx: Path(ff).name for idx, ff in enumerate(folders)}
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.glob('*.png'):
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1

    print("Train model")
    model = svm.SVC()#C=0.1, gamma=0.0001, kernel='poly', degree=2, decision_function_shape='ovo')
    #model = AdaBoostClassifier(n_estimators=50)
    #model = BaggingClassifier(n_estimators=50)  # For sfn classification
    #model = RandomForestClassifier(n_estimators=50)
    #model = GradientBoostingClassifier(n_estimators=50, verbose=1)
    #model = GridSearchCV(svm.SVC(), SVM_PARAM_GRID)
    #model = KNeighborsClassifier(n_neighbors=len(folders))#, weights='distance')
    model.fit(train_x, train_y)
    return model, class_map

def build_class_map(folders):
    return {idx: Path(ff).name for idx, ff in enumerate(folders)}

def train_tf(folders):
    import tensorflow as tf
    class_map = build_class_map(folders)
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.iterdir():
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1
    train_x = np.array(train_x)[..., np.newaxis]
    train_y = tf.one_hot(train_y, len(folders))
    output_types = (tf.uint8, tf.uint8)
    output_shapes = ((TARGET_HEIGHT, TARGET_WIDTH, 1), (len(folders)))
    dataset = tf.data.Dataset.from_generator(lambda: zip(train_x, train_y), output_types=output_types, output_shapes=output_shapes)
    dataset = dataset.shuffle(len(train_y), reshuffle_each_iteration=True)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(16)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1)),
        tf.keras.layers.Conv2D(48, (7, 7), activation='relu'), # 48, (7, 7)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'), # 32, (5, 5)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(24, (3, 3), activation='relu'), # 18, (3, 53)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(folders), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, batch_size=16, epochs=5)
    return model, class_map


def test(model, folders):
    test_x = []
    test_y = []
    samples = 100
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        files = list(folder.glob('*.png'))
        random.shuffle(files)
        for ff in files:
            if idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            test_x.append(arr)
            test_y.append(cidx)
            idx += 1

    pred_y = model.predict(test_x)
    tp_idx = (pred_y == test_y)
    tp = len(pred_y[tp_idx])
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


def test_tf(model, folders):
    test_x = []
    test_y = []
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        files = list(folder.iterdir())
        random.shuffle(files)
        for ff in files:
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            test_x.append(arr)
            test_y.append(cidx)

    test_x = np.array(test_x)[..., np.newaxis]
    test_y = np.array(test_y)
    test_result = []
    batch_size = 16
    for idx in range(0, len(test_x), batch_size):
        data = test_x[idx:idx+batch_size]
        pred = model.predict(data)
        pidx = np.argmax(pred, axis=-1)
        test_result.extend(list(pidx))

    test_result = np.array(test_result)
    tp = test_result[test_result==test_y].size
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


def predict(region, model_name):
    if np.max(region) == 1:
        region *= 255
    m_info = pickle.load(open(f"sklearn_models/{model_name}.model", "rb"))
    model = m_info['model']
    w = m_info['w']
    h = m_info['h']
    region = Image.fromarray(region.astype(np.uint8)).resize((w, h))
    pred = model.predict(np.array([region]))
    return m_info['class_map'][round(pred[0][0])]


def train_rests_above8(filename = "rests_above8.model"):
    folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_rests(filename = "rests.model"):
    folders = ["rest_whole", "rest_quarter", "rest_8th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_all_rests(filename = "all_rests.model"):
    folders = ["rest_whole", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_sfn(filename = "sfn.model"):
    folders = ["sharp", "flat", "natural"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_clefs(filename = "clef.model"):
    folders = ["gclef", "fclef"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_noteheads():
    folders = ["notehead_solid", "notehead_hollow"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(f"notehead.model", "wb"))

if __name__ == "__main__":
    samples = 400
    collect_data(samples=samples)

    # folders = ["gclef", "fclef"]; model_name = "clef"
    # folders = ["sharp", "flat", "natural"]; model_name = "sfn"
    #folders = ["rest_whole", "rest_quarter", "rest_8th"]; model_name = "rests"
    # folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]; model_name = "rests_above8"

    #folders = ['clefs', 'sfns']; model_name = 'clefs_sfns'
    #folders = ["rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd", "rest_64th"]

    # Sklearn model
    #model, class_map = train([f"ds2_dense/train_data/{folder}" for folder in folders])
    #test(model, [f"ds2_dense/test_data/{folder}" for folder in folders])

    # TF-based model
    # model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    # test_tf(model, [f"test_data/{folder}" for folder in folders])

    #output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    #pickle.dump(output, open(f"oemer/sklearn_models/{model_name}.model", "wb"))
