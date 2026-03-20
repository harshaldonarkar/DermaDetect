import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

classes = {
    0: "actinic keratoses and intraepithelial carcinomae(Cancer)",
    1: "basal cell carcinoma(Cancer)",
    2: "benign keratosis-like lesions(Non-Cancerous)",
    3: "dermatofibroma(Non-Cancerous)",
    4: "melanocytic nevi(Non-Cancerous)",
    5: "pyogenic granulomas and hemorrhage(Can lead to cancer)",
    6: "melanoma(Cancer)",
}

INPUT_SHAPE = (224, 224, 3)
MODEL_PATH = "best_model_efficientnet.keras"


def build_model():
    base = EfficientNetB0(weights=None, include_top=False, input_shape=INPUT_SHAPE)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(7, activation="softmax")(x)
    return Model(inputs=base.input, outputs=output)


model = build_model()
if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)
    print(f"Loaded weights from {MODEL_PATH}")
else:
    print(
        f"WARNING: {MODEL_PATH} not found. Run train_model.py to train the model before using the app."
    )
