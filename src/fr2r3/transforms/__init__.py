from fr2r3.transforms.colour import colour
from fr2r3.transforms.fourier import fourier
from fr2r3.transforms.hough import hough

# List of all available transforms.
TRANSFORMS = {
    'colour': colour,
    'fourier': fourier,
    'hough': hough,
}
