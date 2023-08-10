"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
from lrp_methods.filter import relevance_filter
# from filter import relevance_filter

top_k_percent = 0.04  # Proportion of relevance scores that are allowed to pass.


class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class RelevancePropagationAvgPool1d(nn.Module):
    """Layer-wise relevance propagation for 1D average pooling.

    Attributes:
        layer: 1D average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.MaxPool2d, mode: str = "max", eps: float = 1.0e-05) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        # print(f"a.shape: {a.shape}, r.shape: {r.shape}, z.shape: {z.shape}")
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationSeparableConv1D(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Conv2d, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        layers = []
        for i, l in enumerate(layer.modules()):
            if i == 0:
                continue
            print(l)
            # l1 = RelevancePropagationConv1d(l)
            # layers.append(l1)
        self.layers = layers
        print(f"Init finished. Layers: {self.layers}")

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        print(f"Started forward pass, r: {r}, a: {a}")
        r = self.layers[0](a, r)
        r = self.layers[1](a, r)
        return r

class RelevancePropagationConv1d(nn.Module):
    """Layer-wise relevance propagation for 1D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Conv1d, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        # print(f"Initialized conv1d. Layer: {layer}")

        self.layer = layer

        if mode == "z_plus":
            # print("conv1d: before weight")
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            # print("conv1d: before bias")
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        # print("conv1d: before eps")
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # print(f"started forward: a: {a}, r: {r}")
        r = relevance_filter(r, top_k_percent=top_k_percent)
        z = self.layer.forward(a) + self.eps
        # print(f"a.shape: {a.shape}, r.shape: {r.shape}, z.shape: {z.shape}")
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Conv2d, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = relevance_filter(r, top_k_percent=top_k_percent)
        z = self.layer.forward(a) + self.eps
        # print(f"a.shape: {a.shape}, r.shape: {r.shape}, z.shape: {z.shape}")
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Linear, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = relevance_filter(r, top_k_percent=top_k_percent)
        z = self.layer.forward(a) + self.eps
        # print(f"a.shape: {a.shape}, r.shape: {r.shape}, z.shape: {z.shape}")
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # print(f"a.shape: {a.shape}, r.shape before: {r.shape}")
        r = r.view(size=a.shape)
        # print(f"a.shape: {a.shape}, r.shape after: {r.shape}")
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r

class RelevanceZeroPad(nn.Module):
    """Zero pad layer for relevance propagation.

    """

    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # print(self.layer)
        # print("PADDING", self.layer.padding)
        padding = self.layer.padding
        # print(f"a.shape: {a.shape}, r.shape before: {r.shape}")
        left = padding[0]
        right = padding[1]
        up = padding[2]
        down = padding[3]
        # print(f"left: {left}, right: {right}, up: {up}, down: {down}")
        # r = r[:, :, left:-right, up:-down]
        r = r[:, :, up:-down, left:-right]
        # print(f"a.shape: {a.shape}, r.shape after: {r.shape}")
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # print(f"a.shape: {a.shape}, r.shape: {r.shape}")
        return r
