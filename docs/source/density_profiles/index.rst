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

To calculate density profiles with *Cosmic Profiles*, we first instantiate a ``DensProfs`` object called ``cprofiles``, similar to what we saw in :ref:`Shape Estimation section<Shape Estimation>`. Now we can simply invoke the command::

    dens_profs_db = cprofiles.getDensProfsDirectBinning(r_over_r200),

where the float array ``dens_profs_db`` of shape :math:`(N_{\text{obj}}, N_r)` contains the estimated density profiles. This assumes that the float array that specifies for which unitless spherical radii ``r_over_r200`` the local density should be calculated has shape :math:`N_r`. Specifying radial bins with equal spacing in logarithmic space :math:`\log (\delta r/r_{200}) = \mathrm{const}` is common practice.

.. note:: In case of a Gadget-style HDF5 snapshot output, one must additionally specify `obj_type` = `dm` or `gx` to ``getDensProfsDirectBinning()`` in order to have the density profiles calculated for either dark matter halos or galaxies.

As the naming suggests, ``getDensProfsDirectBinning()`` estimates density profiles using a direct-binning approach, i.e. brute-force binning of particles into spherical shells and subsequent counting. On the other hand::

    dens_profs_kb = cprofiles.getDensProfsKernelBased(r_over_r200)

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

.. math:: \rho(r) = \rho_s \exp\left(-\frac{2}{\alpha}\left[\left(\frac{r}{r_{-2}}\right)^{\alpha}-1\right]\right).

Finally, the :math:`\alpha \beta \gamma` density profile (`Zemp et al 2011 <https://arxiv.org/abs/1107.5582>`_) is a generalization of the Navarro-Frank-White (NFW) halo density profile with the parametrization

.. math:: \rho(r) = \frac{\rho_s}{(r/r_s)^{\gamma}[1+(r/r_s)^{\alpha}]^{(\beta-\gamma)/\alpha}}.

To fit density profiles according to model ``method``, a string which can be either 'nfw', 'hernquist', 'einasto' or 'alpha_beta_gamma', invoke the method::

    best_fits = cprofiles.getDensProfsBestFits(dens_profs_fit, r_over_r200_fit, method).

The first argument ``dens_profs_fit`` is an array of shape :math:`(N_{\text{obj}}, N_r)` containing the density profiles defined at radii ``r_over_r200_fit``, possibly obtained via ``getDensProfsDirectBinning()`` or ``getDensProfsDirectBinning()``, with some non-reliable values removed. The last argument ``method`` is 1 of 4 possible strings corresponding to the density profile model, i.e. either ``nfw``, ``hernquist``, ``einasto`` or ``alpha_beta_gamma``. The returned array ``best_fits`` will store the best-fit results and has shape (:math:`N_{\text{obj}}, n`), ``n`` being the number of parameters in model ``method``.

Once density profiles have been fit, concentrations of objects can be calculated, defined as

.. math:: c = \frac{R_{200}}{r_{-2}},

with :math:`r_{-2} = r_s` the characteristic or scale radius of the corresponding density profile model. To this end, invoke::

    cs = cprofiles.getConcentrations(dens_profs_fit, r_over_r200_fit, method),

which will return a float array ``cs`` of shape (:math:`N_{\text{obj}},`).

.. note:: In case of a Gadget-style HDF5 snapshot output, one must additionally specify `obj_type` = `dm` or `gx` to ``getDensProfsBestFits()`` in order to have the density profiles fits for either dark matter halos or galaxies.
