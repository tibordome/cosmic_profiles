Mock Halo Generator
========================

|pic1| |pic2|

.. |pic1| image:: Oblate.png
   :width: 45%

.. |pic2| image:: Prolate.png
   :width: 45%

We provide a mock halo generator in the function ``genAlphaBetaGammaHalo()`` documented in :ref:`the code reference<Cosmic Shapes Code Reference>`. The alpha-beta-gamma density profile is a generalization of the Navarro-Frank-White (NFW) halo density profile with the parametrization

.. math:: \rho(r) = \frac{\rho_0}{(r/r_s)^{\gamma}[1+(r/r_s)^{\alpha}]^{[(\beta-\gamma)/\alpha]}}.

The function ``genAlphaBetaGammaHalo()`` will sample halo particles from the above density profile while the principal axis ratios between `a`, `b` and `c` are assumed to be constant across `r` = :math:`r_{\text{ell}}` = `a`. The following snippet samples a halo from a density distribution with parameters ``rho_0`` = 1, ``r_s`` = 1, ``alpha`` = 1, ``beta`` = 3, ``gamma`` = 1. The larger ``N_bin``, the finer the sampling.

.. code-block:: python

    from cosmic_shapes import genAlphaBetaGammaHalo
    
    # Generate 1 mock halo
    rho_0 = 1; r_s = 1; alpha = 1; beta = 3; gamma = 1
    N_bin = 20; a_max = 2; delta_a = 0.1
    a = np.linspace(delta_a,a_max,N_bin)
    assert np.allclose(a[2]-a[1], delta_a)
    b = a*0.2 
    c = a*0.2 # This will be a prolate halo

    halo_x, halo_y, halo_z = genAlphaBetaGammaHalo(N_MIN, alpha, beta, gamma, rho_0, r_s, a, b, c)
