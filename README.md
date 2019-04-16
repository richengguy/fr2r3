# f: R2 → R3

The way to read the title of this project is, "A function that maps the set of
two-dimensional real numbers to the set of three-dimensional real numbers."
You see this kind of mathematical shorthand used to provide a generic definition
of an image.  It makes sense; an image is either a 2D scalar field (monochrome
images) or a mapping from a 2D space to a 3D space (colour images).

The `fr2r3` application was designed to generate a small set of visualizations
that demonstrates how the exact same image can be viewed in completely different
ways.  Put a little bit more formally, it's possible to define a set of
operators that map the image from one high-dimensional space to another
high-dimensional space.  The specifics depend on the transform, but they are
chosen to pull out something interesting, or useful, about an image.

You can find out more details about the algorithms here: [An Example](docs/)

## Installation

The supported method for installing `fr2r3` is to use
[conda](https://conda.io/en/latest/).  First, make sure that you have conda
installed locally (development was done with miniconda but the full Anaconda
distribution will work as well).  Next, check out the repository from
[GitHub](https://github.com/richengguy/fr2r3),

```
$ git clone https://github.com/richengguy/fr2r3.git
```

Setting up the environment to use `fr2r3` is then simply

```
$ conda env create
$ conda activate fr2r3
```

## Running

The main entrypoint is the `fr2r3` command.  Use `--help` to see what
subcommands are available and how to use them.

```
Usage: fr2r3 [OPTIONS] COMMAND [ARGS]...

  Apply a transform operator onto an image.

  The fr2r3 application will apply a transformation (e.g. Fourier transform)
  onto the provided image.  The transforms are meant for visualization
  purposes and have been modified to be visually pleasing.

Options:
  --help  Show this message and exit.

Commands:
  filter      Apply the transform on to the image.
  ls          List the available transforms.
  synthesize  Generate a synthetic image.
```

## Licence

All source code is licensed under a BSD 3-Clause Licence.  The contents of the
`docs` folder are licensed under a [CC-BY-SA-4.0](http://creativecommons.org/licenses/by-sa/4.0/) licence.
