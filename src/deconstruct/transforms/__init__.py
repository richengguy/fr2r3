from deconstruct.transforms.fourier import fourier
from deconstruct.transforms.hough import hough
from deconstruct.transforms.structure_tensor import structure_tensor
from deconstruct.transforms.wavelet import wavelet

# List of all available transforms.
TRANSFORMS = {
    'fourier': fourier,
    'hough': hough,
    'wavelet': wavelet,
    'structure-tensor': structure_tensor
}
