import base64
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Last convolutional activation layer in EfficientNetB0
LAST_CONV_LAYER = "top_activation"


def generate_gradcam(model, img_array, class_idx):
    """
    Compute Grad-CAM heatmap for the given class index.
    img_array: shape (1, H, W, 3), float32
    Returns a 2D numpy heatmap in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(original_img_array, heatmap, alpha=0.45):
    """
    Blend heatmap over original image.
    original_img_array: (H, W, 3) uint8 RGB
    Returns a base64-encoded PNG data URI string.
    """
    h, w = original_img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    original_uint8 = np.uint8(original_img_array)
    superimposed = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    pil_img = Image.fromarray(superimposed)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def to_base64(pil_img):
    """Convert a PIL image to a base64 PNG data URI."""
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
