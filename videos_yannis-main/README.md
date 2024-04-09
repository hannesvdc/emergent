# Videos of Chimera States

Type-I Chimera in the Stuart-Landau Ensemble
---------

See [this paper](https://arxiv.org/pdf/1905.00218.pdf) for a description of the dynamics.

Using

    python create_histogram_video.py

a sequence of images are created from which a video of the oscillator dynamics can be made
(e.g. by using ffmpeg).
The images show the oscillator values on the right, and the distribution of the oscillator values
in the complex plane on the left.
In addition, the oscillators on the right are colored with their behavior
corresponding to their location on the left, with the color given by the color map below the histogram.

Type-I Chimera in the MCGLE with two spatial dimensions
---------

See [this paper](https://arxiv.org/pdf/1503.04053.pdf) for a description of the dynamics.

Using

    python create_3d_pdf.py

a sequence of images are created from which a video of the dynamics can be made
(e.g. by using ffmpeg).
The images show a 3-dimensional histogram at each time step, where the axis are
  - The absolute value of W at each point in space
  - The real part of the local curvature at each point in space
  - The imaginary part of the local curvature at each point in space

The color of each voxel encodes the number of grid points in each histogram bin,
that is, the number of spatial grid points with this behavior.
