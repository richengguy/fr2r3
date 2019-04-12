from deconstruct.transforms.colour import colour
from deconstruct.transforms.fourier import fourier
from deconstruct.transforms.hough import hough
from deconstruct.transforms.structure_tensor import structure_tensor
from deconstruct.transforms.wavelet import wavelet

# List of all available transforms.
TRANSFORMS = {
    'colour': colour,
    'fourier': fourier,
    'hough': hough,
    'wavelet': wavelet,
    'structure-tensor': structure_tensor
}
