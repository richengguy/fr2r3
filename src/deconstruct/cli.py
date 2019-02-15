import pathlib
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
@click.option('--outdir', '-o', type=click.Path(exists=True, file_okay=False),
              help='Path to output directory.', default='.')
@click.option('--format', '-f', type=click.Choice(['jpg', 'png']),
              help='Output format.', default='png')
@click.argument('fname', metavar='FILTER')
@click.argument('imgname', metavar='IMAGE',
                type=click.Path(dir_okay=False, exists=True))
def filter(outdir, format, fname, imgname):
    '''Apply the transform on to the image.

    The output will
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
    out = tfrm(img)
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


if __name__ == '__main__':
    main()
