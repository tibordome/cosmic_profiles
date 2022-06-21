Density Profiles
=================

|pic1|

.. |pic1| image:: RhoProfFitObj0_015.png
   :width: 60%

We have added density profile estimation capabilities even though the focus of this package is on shapes of halos and galaxies. We contend ourselves with spherically averaged density profiles, which are thus defined with respect to the halocentric radius

.. math:: r = \sqrt{x^2+y^2+z^2},

where :math:`(x,y,z)` are the coordinates of a point cloud particle in some coordinate system centered on either the center of mass or the mode of the cloud. The density profile describes the radial mass distribution of the points in the cloud in units of :math:`M_{\odot}h^2/(\mathrm{Mpc})^3`. 

To calculate density profiles with ``cosmic_shapes``, we first instantiate a ``CosmicShapes`` object called ``cshapes`` as described in the :ref:`Shape Estimation section<Shape Estimation>`. Now we can simply invoke the command::

    rho_profs = cshapes.calcDensProfs(r_over_r200)

which will return a float array ``rho_profs`` of shape :math:`(\text{number of objects}, N)`. This assumes that the float array that specifies for which unitless spherical radii ``r_over_r200`` the local density should be calculated has shape :math:`N`. Specifying radial bins with equal spacing in logarithmic space :math:`\log (\delta r/r_{200}) = \mathrm{const}` is common practice.

.. note:: In case of a Gadget-style HDF5 snapshot output, one must additionally specify `obj_type` = `dm` or `gx` to have the density profiles calculated for either dark matter halos or galaxies.


