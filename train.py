from oemer import train
import sys
import time

def write_text_to_file(text, path):
    with open(path, "w") as f:
        f.write(text)

if len(sys.argv) != 2:
    print("Usage: python train.py <model_name>")
    sys.exit(1)

def get_model_base_name(model_name: str) -> str:
    timestamp = str(round(time.time()))
    return f"{model_name}_{timestamp}"
                    
model = sys.argv[1]

if model == "dense":
    model = train.train_model("ds2_dense", steps=500, epochs=1, win_size=128)
    filename = get_model_base_name("segnet")
    write_text_to_file(model.to_json(), filename + ".json")
    model.save_weights(filename + ".h5")
elif model == "cvc":
    model = train.train_model("CvcMuscima-Distortions", steps=500, epochs=5, win_size=128, data_model="cvc")
    filename = get_model_base_name("unet")
    write_text_to_file(model.to_json(), filename + ".json")
    model.save_weights(filename + ".h5")
else:
    print("Unknown model: " + model)
    sys.exit(1)
