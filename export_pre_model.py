import warnings
import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple
import mobile_sam as SAM
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


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

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

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


if __name__ == '__main__':
    output_names = ['output']

    checkpoint = 'mobile_sam.pt'
    model_type = 'vit_t'
    output_path = 'mobile_sam_preprocess.onnx'
    quantize = False

    # Target image size is 1024x720
    image_size = (1024, 720)

    output_raw_path = output_path
    if quantize:
        # The raw directory can be deleted after the quantization is done
        output_name = os.path.basename(output_path).split('.')[0]
        output_raw_path = '{}/{}_raw/{}.onnx'.format(
            os.path.dirname(output_path), output_name, output_name)
        os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)

    sam = SAM.sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cpu')
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device='cpu')[None]


    class Model(torch.nn.Module):
        def __init__(self, checkpoint, model_type):
            super().__init__()
            self.sam = SAM.sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam.to(device='cpu')
            self.predictor = SAM.SamPredictor(self.sam)

        def forward(self, transformed_image):
            img = transformed_image.permute(0, 3, 1, 2).contiguous()

            self.predictor.set_torch_image(img, img.shape[:2])
            if 'interm_embeddings' not in output_names:
                return self.predictor.get_image_embedding()
            else:
                return self.predictor.get_image_embedding(), torch.stack(self.predictor.interm_features, dim=0)


    dummy_inputs = {
        "transformed_image": input_image_torch,
    }


    model = Model(checkpoint, model_type)
    dynamic_axes={"transformed_image": {0: "batch", 1: "height", 2: "width"}}


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_raw_path, "wb") as f:
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=16,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if quantize:
        quantize_dynamic(
            model_input=output_raw_path,
            model_output=output_path,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
