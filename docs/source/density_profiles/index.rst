Density Profiles
========================

**************************
Density Profile Estimation
**************************

|pic0|

.. |pic0| image:: RhoProfObj0_015.png
   :width: 60%

We have added density profile estimation capabilities. We contend ourselves with spherically averaged density profiles, which are thus defined with respect to the halocentric radius

.. math:: r = \sqrt{x^2+y^2+z^2},

where :math:`(x,y,z)` are the coordinates of a point cloud particle in some coordinate system centered on either the center of mass or the mode of the cloud. The density profile describes the radial mass distribution of the points in the cloud in units of :math:`M_{\odot}h^2/(\mathrm{Mpc})^3`. 

To estimate density profiles with *Cosmic Profiles*, we first instantiate a ``DensProfs`` object called ``cprofiles``, similar to what we saw in the :ref:`Shape Estimation section<Shape Estimation>`. Now we can simply invoke the command::

    dens_profs_db = cprofiles.estDensProfs(r_over_r200, select = [0, nb_objs-1], direct_binning = True, spherical = True),

where the float array ``dens_profs_db`` of shape :math:`(N_{\text{pass}}, N_r)` contains the estimated density profiles. The ``select`` argument expects a list of two integers indicating for which objects to estimate the density profile. In the example above, all objects are considered since the range ``[0, nb_objs-1]`` spans the maximum range of indices. This assumes that the float array that specifies for which unitless spherical radii ``r_over_r200`` the local density should be calculated has shape :math:`N_r`. Specifying radial bins with equal spacing in logarithmic space :math:`\log (\delta r/r_{200}) = \mathrm{const}` is common practice.

.. note:: In case of a Gadget-style HDF5 snapshot output, you may want to additionally specify ``obj_type = 'dm'`` or ``'gx'`` to ``estDensProfs()`` in order to have the density profiles calculated for either dark matter halos or galaxies.

As the naming suggests, with ``direct_binning = True`` we estimate density profiles using a direct-binning approach, i.e. brute-force binning of particles into spherical shells and subsequent counting. The user also has the liberty to invoke an ellipsoidal shell-based density profile estimation algorithm by setting the boolean ``spherical = False``. Note, however, that this necessitates that ``cprofiles`` is an object of the class ``DensShapeProfs`` or ``DensShapeProfsHDF5``, providing access to shape profiling capabilities.

.. note:: If ``spherical = False``, the user also has the discretion to set 2 keyword arguments, namely the booleans ``reduced`` and ``shell_based`` that are explained in the :ref:`Shape Estimation section<Shape Estimation>`.

See `Gonzalez et al. 2022 <https://arxiv.org/abs/2205.06827>`_ for an application of the ellipsoidal shell-based density profile estimation technique. On the other hand, with ``direct_binning = False`` we perform a kernel-based density profile estimation, cf. `Reed et al. 2005 <https://academic.oup.com/mnras/article/357/1/82/1039256>`_. Kernel-based approaches allow estimation of profiles without excessive particle noise.

.. _Density Profile Fitting:

**************************
Density Profile Fitting
**************************

|pic1|

.. |pic1| image:: RhoProfFitObj0_015.png
   :width: 60%

Apart from estimating density profiles using the direct-binning or the kernel-based approach, this package supports density profile fitting assuming a certain density profile model. Four different density profile models can be invoked. First, the NFW-profile (`Navarro et al. <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_) defined by

.. math:: \rho(r) = \frac{\rho_s}{(r/r_s)(1+r/r_s)^2}.

Secondly, the Hernquist profile (`Hernquist 1990 <https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H/abstract>`_) given by

.. math:: \rho(r) = \frac{\rho_s}{(r/r_s)(1+r/r_s)^3}.

Thirdly, the Einasto profile (`Einasto 1965 <https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E/abstract>`_) defined by an additional parameter :math:`\alpha` via

.. math:: \rho(r) = \rho_s \exp\left(-\frac{2}{\alpha}\left[\left(\frac{r}{r_s}\right)^{\alpha}-1\right]\right).

Finally, the :math:`\alpha \beta \gamma` density profile (`Zemp et al 2011 <https://arxiv.org/abs/1107.5582>`_) is a generalization of the Navarro-Frank-White (NFW) halo density profile with the parametrization

.. math:: \rho(r) = \frac{\rho_s}{(r/r_s)^{\gamma}[1+(r/r_s)^{\alpha}]^{(\beta-\gamma)/\alpha}}.

To fit density profiles according to model ``method``, a string which can be either ``nfw``, ``hernquist``, ``einasto`` or ``alpha_beta_gamma``, invoke the method::

    best_fits = cprofiles.fitDensProfs(dens_profs_fit, r_over_r200_fit, method, select = [0, nb_objs-1]).

The first argument ``dens_profs_fit`` is an array of shape :math:`(N_{\text{pass}}, N_r)` containing the density profiles defined at radii ``r_over_r200_fit``, possibly obtained via ``getDensProfsDirectBinning()`` or ``getDensProfsDirectBinning()``, with some non-reliable values removed. The last argument ``method`` is 1 of 4 possible strings corresponding to the density profile model, i.e. either ``nfw``, ``hernquist``, ``einasto`` or ``alpha_beta_gamma``. The returned array ``best_fits`` will store the best-fit results and has shape (:math:`N_{\text{pass}}, n`), :math:`n` being the number of parameters in model ``method``.

Once density profiles have been fit, concentrations of objects can be calculated, defined as

.. math:: c = \frac{R_{200}}{r_s},

with :math:`r_s` the characteristic or scale radius of the corresponding density profile model. To this end, invoke::

    cs = cprofiles.estConcentrations(dens_profs_fit, r_over_r200_fit, method, select = [0, nb_objs-1]),

which will return a float array ``cs`` of shape (:math:`N_{\text{pass}},`).

.. note:: In case of a Gadget-style HDF5 snapshot output, you may want to additionally specify ``obj_type = 'dm'`` or ``'gx'`` to ``getDensProfsBestFits()`` in order to have the density profiles fits for either dark matter halos or galaxies.

The density profiles, for instance ``dens_profs_db``, and their fits can be visualized using::

    cprofiles.plotDensProfs(dens_profs_db, r_over_r200, dens_profs_fit, r_over_r200_fit, method, VIZ_DEST, select = [0, nb_objs-1])

where ``dens_profs_fit`` and ``r_over_r200_fit`` refer to those estimated density profile values that the user would like the fitting operation to be carried out over, e.g. ``dens_profs_fit = dens_profs_db[:,25:]`` and ``r_over_r200_fit = r_over_r200[25:]`` to discard the values that correspond to deep layers of halos/galaxies/objects. Typically, the gravitational softening scale times some factor and / or information from the local relaxation timescale is used to estimate the inner convergence radius. For guidance on choosing the inner convergence radius see `Navarro et al 2010 <https://academic.oup.com/mnras/article/402/1/21/1028856>`_.
