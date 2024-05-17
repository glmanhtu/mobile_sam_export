from typing import Tuple

import cv2
import onnx
import onnxruntime
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from export_pre_model import ResizeLongestSide


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


transform = ResizeLongestSide(1024)

image = cv2.imread('picture2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_image = transform.apply_image(image)
input_image_torch = torch.as_tensor(input_image, device='cpu')
input_image_torch = input_image_torch[None]

ort_inputs = {
    "transformed_image": input_image_torch.numpy(),
}

onnx_pre_model_path = "mobile_sam_preprocess.onnx"
onnx_model = onnx.load(onnx_pre_model_path)
ort_pre_session = onnxruntime.InferenceSession(onnx_pre_model_path)
features = ort_pre_session.run(None, ort_inputs)[0]


input_point = np.array([[250, 375]])
input_label = np.array([1])
onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": features,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

onnx_model_path = "mobile_sam.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

masks, im_masks, _, low_res_logits = ort_session.run(None, ort_inputs)

plt.figure(figsize=(10,10))
plt.imshow(image)

masks = masks > 0
show_mask(masks[0], plt.gca())
# plt.gca().imshow(im_masks[0])
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
