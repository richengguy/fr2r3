import pathlib
import sys

import click

from fr2r3.synthesize import SYNTHETIC_IMAGES
from fr2r3.transforms import TRANSFORMS
from fr2r3.io import imread, to_ndarray, to_pil_image


@click.group()
def main():
    '''Apply a transform operator onto an image.

    The fr2r3 application will apply a transformation (e.g. Fourier transform)
    onto the provided image.  The transforms are meant for visualization
    purposes and have been modified to be visually pleasing.
    '''


@main.command()
def ls():
    '''List the available transforms.'''
    click.secho('Transforms:', bold=True)
    for k in TRANSFORMS:
        click.echo(' - %s' % k)

    click.secho('Synthetic Images:', bold=True)
    for k in SYNTHETIC_IMAGES:
        click.echo(' - %s' % k)


@main.command()
@click.option('--outdir', '-o', type=click.Path(exists=True, file_okay=False),
              help='Path to output directory.', default='.')
@click.option('--format', '-f', type=click.Choice(['jpg', 'png']),
              help='Output format.', default='png')
@click.option('--scale', '-s', type=click.FLOAT, default=1.0,
              help='Scale the image prior to processing.')
@click.argument('fname', metavar='FILTER')
@click.argument('imgname', metavar='IMAGE',
                type=click.Path(dir_okay=False, exists=True))
def filter(outdir, format, scale, fname, imgname):
    '''Apply the transform on to the image.

    The 'filter' command will generate a new image, roughly the same size as
    the original.  The exact output will depend on the filter, but in general,
    each filter applies some sort of transform on the image and then visualizes
    the result.  The list of available filters can be obtained through the 'ls'
    command.
    '''
    try:
        tfrm = TRANSFORMS[fname]
    except KeyError:
        click.secho('Error: ', bold=True, fg='red', nl=False)
        click.echo('Unknown filter "%s"' % fname)
        sys.exit(-1)

    outdir = pathlib.Path(outdir)

    click.echo('Applying %s...' % click.style(fname, fg='blue'))
    img = to_ndarray(imread(imgname))
    out = tfrm(img, scale=scale)
    if isinstance(out, tuple):
        for i, elem in enumerate(out):
            outpath = outdir / ('%s-%d.%s' % (fname, i, format))
            elem = to_pil_image(elem)
            elem.save(outpath)
            click.echo('-- Saved %s' % click.style(str(outpath), fg='blue'))
    else:
        out = to_pil_image(out)
        outpath = outdir / ('%s.%s' % (fname, format))
        out.save(outpath)
        click.echo('-- Saved %s' % click.style(str(outpath), fg='blue'))


@main.command()
@click.option('--outdir', '-o', type=click.Path(exists=True, file_okay=False),
              help='Path to output directory.', default='.')
@click.option('--format', '-f', type=click.Choice(['jpg', 'png']),
              help='Output format.', default='png')
@click.option('--size', type=click.INT, default=(512, 512), nargs=2,
              metavar='WIDTH HEIGHT', help='Output size.')
@click.argument('imgtype', metavar='TYPE')
@click.argument('args', metavar='ARGS', nargs=-1, type=click.FLOAT)
def synthesize(outdir, format, size, imgtype, args):
    '''Generate a synthetic image.

    The 'synthesize' command is used to generate a synthetic image with some
    well-defined properties.  The list of availble image types can be obtained
    using the 'ls' command.
    '''
    try:
        method = SYNTHETIC_IMAGES[imgtype]
    except KeyError:
        click.secho('Error: ', bold=True, fg='red', nl=False)
        click.echo('Unknown image type "%s"' % imgtype)
        sys.exit(-1)

    outdir = pathlib.Path(outdir)

    click.echo('Generating %s...' % click.style(imgtype, fg='blue'))
    out = method(size, *args)
    out = to_pil_image(out)
    outpath = outdir / ('%s.%s' % (imgtype, format))
    out.save(outpath)
    click.echo('-- Saved %s' % click.style(str(outpath), fg='blue'))


if __name__ == '__main__':
    main()
