import torch
import torch.nn as nn

class SigmoidF64(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

model = SigmoidF64().eval().double()
dummy = torch.ones((3, 4, 5), dtype=torch.float64)

torch.onnx.export(
    model,
    dummy,
    "../models/f64_sigmoid",
    input_names=["x"],
    output_names=["y"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes=None,
)