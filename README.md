# Mobile SAM model exporter for onnxruntime



### Compatibility
The dependencies can be installed by running the following command:
```bash
pip install -r requirements.txt
```

### Export the model
Step 1: Download Mobile SAM pretrained model from https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt

Step 2: Run the script to export feature-extracting model:
```bash
python3 export_pre_model.py
```

Step 3: Run the script to export mask-detecting model:
```bash
python3 export_model.py
```

### Test the exported models
Perform segmentation:
```bash
python3 predict.py
```