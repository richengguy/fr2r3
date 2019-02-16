from deconstruct.transforms.fourier import fourier
from deconstruct.transforms.corner import corner
from deconstruct.transforms.hough import hough

# List of all available transforms.
TRANSFORMS = {
    'fourier': fourier,
    'corner': corner,
    'hough': hough
}
