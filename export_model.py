import torch
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel
import warnings

checkpoint = "mobile_sam.pt"
model_type = "vit_t"

onnx_model_path = "mobile_sam.onnx"


class Model(torch.nn.Module):
    def __init__(self, checkpoint, model_type):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.onnx_model = SamOnnxModel(sam, return_single_mask=True)

    def forward(self, *args, **kwargs):
        (masks, iou_predictions, low_res_masks) = self.onnx_model(*args, **kwargs)
        h, w = masks.shape[-2:]
        mask = masks > 0
        mask_image = mask.reshape(-1, h, w, 1) * 255

        return masks, mask_image.type(torch.uint8), iou_predictions, low_res_masks


sam = Model(checkpoint, model_type)


dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

embed_dim = sam.onnx_model.model.prompt_encoder.embed_dim
embed_size = sam.onnx_model.model.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}
output_names = ["masks", "im_masks", "iou_predictions", "low_res_masks"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            sam,
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