from typing import Tuple

import cv2
import onnx
import onnxruntime
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


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

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


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
