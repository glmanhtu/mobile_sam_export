# Code formatter may move "import segment_anything as SAM" and "import mobile_sam as SAM" to the top
# But this may bring errors after switching models
import warnings

import torch
import numpy as np
import os
from segment_anything.utils.transforms import ResizeLongestSide
import mobile_sam as SAM
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

output_names = ['output']

# # Download Mobile-SAM model "mobile_sam.pt" from https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt
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
