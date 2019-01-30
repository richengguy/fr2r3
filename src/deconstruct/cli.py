import sys

import click

from deconstruct.transforms import TRANSFORMS
from deconstruct.io import imread, to_ndarray, to_pil_image


@click.group()
def main():
    '''Apply some sort of transformation onto an image.

    The photo-deconstruct application will apply a transformation (e.g.
    Fourier) onto the provided image.  The transforms are meant for
    visualization purposes. Therefore, the results are modified in such a way
    that they are pleasant to view.
    '''


@main.command()
def ls():
    '''List the available transforms.'''
    click.secho('Transforms:', bold=True)
    for k in TRANSFORMS:
        click.echo(' - %s' % k)


@main.command()
@click.argument('fname', metavar='FILTER')
@click.argument('imgname', metavar='IMAGE',
                type=click.Path(dir_okay=False, exists=True))
def filter(fname, imgname):
    '''Apply the transform on to the image.'''
    try:
        tfrm = TRANSFORMS[fname]
    except KeyError:
        click.secho('Error: ', bold=True, fg='red', nl=False)
        click.echo('Unknown filter "%s"' % fname)
        sys.exit(-1)

    img = to_ndarray(imread(imgname))
    out = tfrm(img)
    if isinstance(out, tuple):
        for i, elem in enumerate(out):
            elem = to_pil_image(elem)
            elem.save('output-%d.png' % i)
    else:
        out = to_pil_image(out)
        out.save('output.png')


if __name__ == '__main__':
    main()
