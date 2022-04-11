Shape Estimation
=================

***************
Shape Profiles
***************

Shape profiles depict the ellipsoidal shape of a point cloud as a function of the ellipsoidal radius

.. math:: r_{\text{ell}} = \sqrt{x_{\text{pf}}^2+\frac{y_{\text{pf}}^2}{(b/a)^2}+\frac{z_{\text{pf}}^2}{(c/a)^2}},

where :math:`(x_{\text{pf}},y_{\text{pf}},z_{\text{pf}})` are the coordinates of a point cloud particle in the eigenvector coordinate system of the ellipsoid (= principal frame), i.e., :math:`r_{\text{ell}}` corresponds to the semi-major axis :math:`a` of the ellipsoidal surface through that particle.

The shape as a function of ellipsoidal radius can be described by the axis ratios

.. math:: q = \frac{b}{r_{\text{ell}}}, \ \ s = \frac{c}{r_{\text{ell}}},

where :math:`b` and :math:`c` are the eigenvalues corresponding to the intermediate and minor axes, respectively. The ratio of the minor-to-major axis :math:`s` has traditionally been used as a canonical measure of the distribution's sphericity. The axis ratios can be computed from the shape tensor :math:`S_{ij}`, which is the second moment of the mass distribution divided by the total mass:

.. math:: S_{ij} = \frac{1}{\sum_k m_k} \sum_k m_k r^{\text{COM}}_{k,i}r^{\text{COM}}_{k,j}.

Here, :math:`m_k` is the mass of the :math:`k`-th particle, and :math:`r^{\text{COM}}_{k,i}` is the :math:`i`-th component of its position vector with respect to the distribution's center of mass (COM).

To calculate shape profiles with `cosmic_shapes`, we first instantiate a `CosmicShapes` object. Let us assume we are dealing with

* a Gadget-style HDF5 snapshot output containing particle and halo/subhalo data in folders `path/to/folder/snapdir_x` and `path/to/folder/groups_x` with `x` typically a three-digit snapshot number identifier such as '042', respectively. Then we will define an object via::

    cshapes = CosmicShapesGadgetHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, CAT_DEST, VIZ_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_DM_PTCS, MIN_NUMBER_STAR_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, SAFE, withVDisp, start_time)

with arguments explained in :ref:`the code reference<Cosmic Shapes Code Reference>`.

* a very general assortment of point clouds. There is no requirement on the nature of the point clouds whatsoever, yet the shape determination algorithm will perform better the closer the point clouds are to being truly ellipsoidal. Often, the process of identifying such point clouds in a simulation can be challenging, which is why we provide an :ref:`interface<AHF interface>` to the 'Amiga Halo Finder' (AHF) via `pynbody`. For now, we assume that we have identified the point clouds already and thus have the indices of the particles belonging to the point clouds at our disposal::
    
    cshapes = CosmicShapesDirect(dm_xyz, mass_array, h_indices, r_vir, CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_DM_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, SAFE, start_time)

`h_indices` consists of point cloud indices that supply to the constructor of `CosmicShapesDirect`.

.. note:: In case of a Gadget-style HDF5 snapshot output, we have to invoke :math:`cshapes.createCatDM()` before calculating the shape catalogue! This ensures that we extract the halo catalogue from the FoF/SH data.

To calculate the halo shape catalogue, we can invoke the command::

    cshapes.createCatMajorCOMDM(withOvrl)

which will calculate and store the morphological information in `CAT_DEST`. We consider a halo shape determination at a specific :math:`r_{\text{ell}}` to be converged if the fractional difference between consecutive eigenvalue fractions falls below `M_TOL` and the maximum number of iterations `N_WALL` is not yet achieved. If in addition the halo shape profile converges at the radius of :math:`R_{200}` (200-overdensity radius), the shape profile is determined successfully. The :math:`N_{\text{conv}}` shape profiles are then grouped together and dumped as 1D and 2D arrays. The output consists of

* `d_local_dm_x.txt` (`x` being the snap string `SNAP`) of shape (:math:`N_{\text{conv}}`, `D_BINS` + 1): ellipsoidal radii
* `q_local_dm_x.txt` of shape (:math:`N_{\text{conv}}`, `D_BINS` + 1): q shape parameter
* `s_local_dm_x.txt` of shape (:math:`N_{\text{conv}}`, `D_BINS` + 1): s shape parameter
* `minor_local_dm_x.txt` of shape (:math:`N_{\text{conv}}`, `D_BINS` + 1, 3): minor axes vs :math:`r_{\text{ell}}`
* `intermediate_local_dm_x.txt` of shape (:math:`N_{\text{conv}}`, `D_BINS` + 1, 3): intermediate axes vs :math:`r_{\text{ell}}`
* `major_local_dm_x.txt` of shape (:math:`N_{\text{conv}}`, `D_BINS` + 1, 3): major axes vs :math:`r_{\text{ell}}`
* `h_cat_local_x.txt` of length :math:`len(h_{\text{indices}})`: list of lists of indices of converged shape profiles, empty list entry [] for each non-converged halo
* `m_dm_x.txt` of shape (:math:`len(h_{\text{indices}})`,): masses of halos
* `coms_dm_x.txt` of shape (:math:`len(h_{\text{indices}})`,3): CoMs of halos

***************
Overall Shapes
***************

Instead of shape profiles one might also be interested in obtaining the shape parameters and principal axes of the point clouds as a whole. This information is dumped on request if the `withOvrl` argument of `createCatMajorCOMDM()` is set to `True`. In that case, additional output will be added to `CAT_DEST`:

* `d_overall_dm_x.txt` (`x` being the snap string `SNAP`) of shape (:math:`N_{\text{pass}}`,): ellipsoidal radii
* `q_overall_dm_x.txt` of shape (:math:`N_{\text{pass}}`,): q shape parameter
* `s_overall_dm_x.txt` of shape (:math:`N_{\text{pass}}`,): s shape parameter
* `minor_overall_dm_x.txt` of shape (:math:`N_{\text{pass}}`, 3): minor axis
* `intermediate_overall_dm_x.txt` of shape (:math:`N_{\text{pass}}`, 3): intermediate axis
* `major_overall_dm_x.txt` of shape (:math:`N_{\text{pass}}`, 3): major axis
* `h_cat_overall_x.txt` of length :math:`len(h_{\text{indices}})`: list of lists of indices of converged shape profiles, empty list entry [] if halo resolution is too low

.. note:: :math:`N_{\text{pass}}` denotes the number of halos that pass the `MIN_NUMBER_DM_PTCS`-threshold. In other words, if the overall shape determination does not converge, it will appear as NaNs in the output.

*************************************
Velocity Dispersion Tensor Eigenaxes
*************************************

For Gadget-style HDF5 snapshot outputs one can calculate the velocity dispersion tensor eigenaxes by simply activating the boolean `withVDisp` in the constructor of `CosmicShapesGadgetHDF5`. In that case, additional output will be added to `CAT_DEST`, reflecting the velocity-related morphological information at :math:`r_{\text{ell}}` = :math:`R_{200}`:

* `q_vdisp_dm_x.txt` of shape (:math:`N_{\text{pass}}`,): q shape parameter
* `s_vdisp_dm_x.txt` of shape (:math:`N_{\text{pass}}`,): s shape parameter
* `major_overall_dm_x.txt` of shape (:math:`N_{\text{pass}}`, 3): major axis
* `h_cat_overall_x.txt` of length :math:`len(h_{\text{indices}})`: list of lists of indices of converged shape profiles, empty list entry [] if halo resolution is too low


