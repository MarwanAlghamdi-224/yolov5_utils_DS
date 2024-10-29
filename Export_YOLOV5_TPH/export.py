import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.torch_utils import select_device
from models.yolo import Detect

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[0]
        boxes = x[:, :, :4]
        objectness = x[:, :, 4:5]
        scores, classes = torch.max(x[:, :, 5:], 2, keepdim=True)
        scores *= objectness
        classes = classes.float()
        return boxes, scores, classes

def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

def yolov5_export(weights, device, inplace=False, fuse=True):
    model = attempt_load(weights, map_location=device, inplace=inplace, fuse=fuse)
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = False
            m.export = True
    return model

def export_to_onnx(model, img_size, batch_size, output_file, opset_version, dynamic, simplify):
    dynamic_axes = {
        'input': {0: 'batch'},
        'boxes': {0: 'batch'},
        'scores': {0: 'batch'},
        'classes': {0: 'batch'}
    } if dynamic else None

    device = next(model.parameters()).device
    onnx_input_im = torch.zeros(batch_size, 3, *img_size).to(device)

    print('\nExporting the model to ONNX...')
    torch.onnx.export(
        model,
        onnx_input_im,
        output_file,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'classes'],
        dynamic_axes=dynamic_axes
    )

    if simplify:
        print('Simplifying the ONNX model...')
        import onnxsim
        model_onnx = onnx.load(output_file)
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'Simplification check failed'
        onnx.save(model_onnx, output_file)

    print(f'Done: {output_file}')

def main(args):
    suppress_warnings()

    print(f'Starting with weights: {args.weights}')

    device = select_device(args.device)
    model = yolov5_export(args.weights, device, inplace=True)

    model = nn.Sequential(model, DeepStreamOutput())
    img_size = args.size * 2 if len(args.size) == 1 else args.size

    if img_size == [640, 640] and args.p6:
        img_size = [1280, 1280]

    output_file = os.path.basename(args.weights).replace('.pt', '.onnx')
    export_to_onnx(model, img_size, args.batch, output_file, args.opset, args.dynamic, args.simplify)

def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv5 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--p6', action='store_true', help='P6 model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    parser.add_argument('--device', default='cpu', help='Device for inference (default: cpu)')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))