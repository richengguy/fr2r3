from deconstruct.transforms.colour import colour
from deconstruct.transforms.fourier import fourier
from deconstruct.transforms.hough import hough

# List of all available transforms.
TRANSFORMS = {
    'colour': colour,
    'fourier': fourier,
    'hough': hough,
}
