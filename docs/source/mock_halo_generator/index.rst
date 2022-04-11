Mock Halo Generator
========================

We provide a mock halo generator in the function `createHaloFixedAxisRatioRhoAlphaBetaGamma()` documented in :ref:`the code reference<Cosmic Shapes Code Reference>`. The alpha-beta-gamma density profile is a generalization of the Navarro-Frank-White (NFW) halo density profile with the parametrization

.. math:: \rho(r) = \frac{\rho_0}{(r/r_s)^{\gamma}[1+(r/r_s)^{\alpha}]^{[(\beta-\gamma)/\alpha]}}.

The function `createHaloFixedAxisRatioRhoAlphaBetaGamma()` will sample halo particles from the above density profile while the principal axis ratios between `a`, `b` and `c` are assumed to be constant across :math:`r_{\text{ell}}` = `a`.
