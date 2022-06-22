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

To calculate density profiles with ``cosmic_shapes``, we first instantiate a ``CosmicShapes`` object called ``cshapes`` as described in the :ref:`Shape Estimation section<Shape Estimation>`. Now we can simply invoke the command::

    cshapes.calcDensProfsDirectBinning(r_over_r200)

whose output is stored in ``CAT_DEST`` and can be retrieved via ``cshapes.fetchDensProfsDirectBinning()``. The latter will return a float array ``rho_profs`` of shape :math:`(N_{\text{obj}}, N_r)`. This assumes that the float array that specifies for which unitless spherical radii ``r_over_r200`` the local density should be calculated has shape :math:`N_r`. Specifying radial bins with equal spacing in logarithmic space :math:`\log (\delta r/r_{200}) = \mathrm{const}` is common practice.

.. note:: In case of a Gadget-style HDF5 snapshot output, one must additionally specify `obj_type` = `dm` or `gx` to ``calcDensProfsDirectBinning()`` in order to have the density profiles calculated for either dark matter halos or galaxies.

The invokation of ``cshapes.calcDensProfsDirectBinning()`` dumps the following two files to ``CAT_DEST``:

* ``dens_profs_db_x.txt`` of shape (:math:`N_{\text{obj}}, N_r`): density profile values
* ``r_over_r200_db_x.txt`` of shape (:math:`N_r`,): spherical radii at which density profiles are calculated

As the naming suggests, ``calcDensProfsDirectBinning()`` estimates density profiles using a direct-binning approach, i.e. brute-force binning of particles into spherical shells and subsequent counting. On the other hand::

    cshapes.calcDensProfsKernelBased(r_over_r200)

performs a kernel-based density profile estimation, cf. `Reed et al. 2005 <https://academic.oup.com/mnras/article/357/1/82/1039256>`_. Kernel-based approaches allow estimation of profiles without excessive particle noise. 


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

.. math:: \rho(r) = \rho_s \exp\left(-\frac{2}{\alpha}\left[\left(\frac{r}{r_2}\right)^{\alpha}-1\right]\right).

Finally, the :math:`\alpha \beta \gamma` density profile (`Zemp et al 2011 <https://arxiv.org/abs/1107.5582>`_) is a generalization of the Navarro-Frank-White (NFW) halo density profile with the parametrization

.. math:: \rho(r) = \frac{\rho_s}{(r/r_s)^{\gamma}[1+(r/r_s)^{\alpha}]^{(\beta-\gamma)/\alpha}}.

To fit density profiles according to model ``method``, a string which can be either 'nfw', 'hernquist', 'einasto' or 'alpha_beta_gamma', invoke the method::

    cshapes.fitDensProfs(dens_profs, ROverR200, cat, r200s, method = 'einasto').

The first argument ``dens_profs`` is an array of shape :math:`(N_{\text{obj}}, N_r)` containing the density profiles defined at radii ``ROverR200``, possibly obtained via ``calcDensProfsDirectBinning()`` or ``calcDensProfsKernelBased()``. The catalogue information in ``cat`` and ``r200s`` should correspond to ``dens_profs`` in the sense that the number of non-empty lists in ``cat`` matches :math:`N_{\text{obj}}` exactly.

The invokation of ``cshapes.fitDensProfs()`` dumps the following two files to ``CAT_DEST``

*  ``dens_prof_best_fits_method_x.txt``, with ``method`` 1 of 4 possible strings, of shape (:math:`N_{\text{obj}}, n`), ``n`` being the number of parameters in model ``method``: best-fit values
*  ``best_fits_r_over_r200_method_x.txt`` of shape (:math:`N_r`,): spherical radii that were used to find best-fit values

The best-fit values can be retrieved via ``cshapes.fetchDensProfsBestFits()``.

