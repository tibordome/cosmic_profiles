.. _Shape Estimation:

Local and Global Shape Estimation
=================================

|pic1| |pic2|

.. |pic1| image:: FDM_1E22HaloT_032.png
   :width: 45%

.. |pic2| image:: FDM_2E21FullHaloTCount_024.png
   :width: 45%

***************
Shape Profiles
***************

Shape profiles depict the ellipsoidal shape of a point cloud as a function of the ellipsoidal radius

.. math:: r_{\text{ell}} = \sqrt{x_{\text{pf}}^2+\frac{y_{\text{pf}}^2}{(b/a)^2}+\frac{z_{\text{pf}}^2}{(c/a)^2}},

where :math:`(x_{\text{pf}},y_{\text{pf}},z_{\text{pf}})` are the coordinates of a point cloud particle in the eigenvector coordinate system of the ellipsoid (= principal frame), i.e., :math:`r_{\text{ell}}` corresponds to the semi-major axis :math:`a` of the ellipsoidal surface through that particle.

The shape as a function of ellipsoidal radius can be described by the axis ratios

.. math:: q = \frac{b}{r_{\text{ell}}}, \ \ s = \frac{c}{r_{\text{ell}}},

where :math:`b` and :math:`c` are the eigenvalues corresponding to the intermediate and minor axes, respectively. The ratio of the minor-to-major axis :math:`s` has traditionally been used as a canonical measure of the distribution's sphericity. A frequently quoted shape parameter is the triaxiality

.. math:: T = \frac{1-q^2}{1-s^2},

which measures the prolateness/oblateness of a halo. :math:`T = 1` describes a completely prolate halo, while :math:`T = 0` describes a completely oblate halo. Halos with :math:`0.33 < T < 0.67` are said to be triaxial. The axis ratios can be computed from the shape tensor :math:`S_{ij}`, which is the second moment of the mass distribution divided by the total mass:

.. math:: S_{ij} = \frac{1}{\sum_k m_k} \sum_k m_k r^{\text{center}}_{k,i}r^{\text{center}}_{k,j}.

Here, :math:`m_k` is the mass of the :math:`k`-th particle, and :math:`r^{\text{center}}_{k,i}` is the :math:`i`-th component of its position vector with respect to the distribution's center (either mode or center of mass).

To calculate shape profiles with *Cosmic Profiles*, we first instantiate a ``CosmicProfiles`` object. Let us assume we are dealing with

* a Gadget-style HDF5 snapshot output containing particle and halo/subhalo data in folders ``path/to/folder/snapdir_x`` and ``path/to/folder/groups_x`` with ``x`` typically a three-digit snapshot number identifier such as '042', respectively. Then we will define an object via::

    from cosmic_profiles import CosmicProfilesGadgetHDF5
    import time
    
    # Parameters
    L_BOX = np.float32(10) # cMpc/h
    CAT_DEST = "/path/to/cat"
    HDF5_GROUP_DEST = "/path/to/groups_035"
    HDF5_SNAP_DEST = "/path/to/snapdir_035"
    VIZ_DEST = "/path/to/viz"
    SNAP_MAX = 16
    D_LOGSTART = -2
    D_LOGEND = 1
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    M_TOL = np.float32(1e-2)
    N_WALL = 100
    N_MIN = 10
    SNAP = '035'
    CENTER = 'mode'
    MIN_NUMBER_DM_PTCS = 200
    MIN_NUMBER_STAR_PTCS = 100
    start_time = time.time()

    # Instantiate object
    cprofiles = CosmicProfilesGadgetHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, CAT_DEST, VIZ_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_DM_PTCS, MIN_NUMBER_STAR_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)

with arguments explained in :ref:`the code reference<Cosmic Profiles Code Reference>`.

* a very general assortment of point clouds. There is no requirement on the nature of the point clouds whatsoever, yet the shape determination algorithm will perform better the closer the point clouds are to being truly ellipsoidal. Often, the process of identifying such point clouds in a simulation can be challenging, which is why we provide an :ref:`interface<AHF interface>` to the 'Amiga Halo Finder' (AHF) via ``pynbody``. For now, we assume that we have identified the point clouds already and that ``obj_indices`` stores the indices of the particles belonging to the point clouds::
    
    from cosmic_profiles import CosmicProfilesDirect
    import time
    
    # Parameters
    xyz = ... # application-dependent
    mass_array = ... # application-dependent
    obj_indices = ... # application-dependent
    r_vir = ... # application-dependent
    CAT_DEST = "/path/to/cat"
    VIZ_DEST = "/path/to/viz"
    SNAP = '035'
    L_BOX = np.float32(10) # cMpc/h
    D_LOGSTART = -2
    D_LOGEND = 1
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    M_TOL = np.float32(1e-2)
    N_WALL = 100
    N_MIN = 10
    SNAP = '035'
    CENTER = 'mode'
    MIN_NUMBER_PTCS = 200
    start_time = time.time()

    # Instantiate object
    cprofiles = CosmicProfilesDirect(xyz, mass_array, obj_indices, r_vir, CAT_DEST, VIZ_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, start_time)

.. note:: In case of a Gadget-style HDF5 snapshot output, we have to invoke ``cprofiles.loadDMCat()`` before calculating the shape catalogue! This ensures that we extract the halo catalogue from the FoF/SH data.

To calculate the local (i.e. as a function of :math:`r_{\text{ell}}`) halo shape catalogue, we can invoke the command::

    cprofiles.calcLocalShapes()

which will calculate and store the morphological information in ``CAT_DEST``. We consider a halo shape determination at a specific :math:`r_{\text{ell}}` to be converged if the fractional difference between consecutive eigenvalue fractions falls below ``M_TOL`` and the maximum number of iterations ``N_WALL`` is not yet achieved. If in addition the halo shape profile converges at the radius of :math:`R_{200}` (200-overdensity radius), the shape profile is determined successfully. The :math:`N_{\text{conv}}` shape profiles are then grouped together and dumped as 1D and 2D arrays. The output consists of

* ``d_local_x.txt`` (``x`` being the snap string ``SNAP``) of shape (:math:`N_{\text{conv}}`, ``D_BINS`` + 1): ellipsoidal radii
* ``q_local_x.txt`` of shape (:math:`N_{\text{conv}}`, ``D_BINS`` + 1): q shape parameter
* ``s_local_x.txt`` of shape (:math:`N_{\text{conv}}`, ``D_BINS`` + 1): s shape parameter
* ``minor_local_x.txt`` of shape (:math:`N_{\text{conv}}`, ``D_BINS`` + 1, 3): minor axes vs :math:`r_{\text{ell}}`
* ``inter_local_x.txt`` of shape (:math:`N_{\text{conv}}`, ``D_BINS`` + 1, 3): intermediate axes vs :math:`r_{\text{ell}}`
* ``major_local_x.txt`` of shape (:math:`N_{\text{conv}}`, ``D_BINS`` + 1, 3): major axes vs :math:`r_{\text{ell}}`
* ``cat_local_x.txt`` of length :math:`N_{\text{conv}}`: list of lists of indices of converged shape profiles, empty list entry [] for each non-converged halo
* ``m_x.txt`` of shape (:math:`N_{\text{conv}}`,): masses of halos
* ``centers_x.txt`` of shape (:math:`N_{\text{conv}}`,3): centers of halos

.. note:: In case of a Gadget-style HDF5 snapshot output, specify ``cprofiles.calcLocalShapesDM()`` to calculate local halo shapes and ``cprofiles.calcLocalShapesGx()`` to calculate local galaxy shapes. The suffix of the output files will be modified accordingly to e.g. ``d_local_dm_x.txt`` or ``d_local_gx_x.txt``, respectively.

***************
Global Shapes
***************

Instead of shape profiles one might also be interested in obtaining the shape parameters and principal axes of the point clouds as a whole. This information is dumped on request by calling::

    cprofiles.calcGlobalShapes(). 

In that case, additional output will be added to ``CAT_DEST``:

* ``d_global_x.txt`` (``x`` being the snap string ``SNAP``) of shape (:math:`N_{\text{pass}}`,): ellipsoidal radii
* ``q_global_x.txt`` of shape (:math:`N_{\text{pass}}`,): q shape parameter
* ``s_global_x.txt`` of shape (:math:`N_{\text{pass}}`,): s shape parameter
* ``minor_global_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): minor axis
* ``inter_global_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): intermediate axis
* ``major_global_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): major axis
* ``cat_global_x.txt`` of length :math:`N_{\text{pass}}`: list of lists of indices of converged shape profiles, empty list entry [] if halo resolution is too low
* ``m_x.txt`` of shape (:math:`N_{\text{pass}}`,): masses of halos
* ``centers_x.txt`` of shape (:math:`N_{\text{pass}}`,3): centers of halos

Again, invoke ``cprofiles.calcGlobalShapesDM()`` to calculate global halo shapes and ``cprofiles.calcGlobalShapesGx()`` to calculate global galaxy shapes, with suffixes adapted accordingly.

.. note:: :math:`N_{\text{pass}}` denotes the number of halos that pass the ``MIN_NUMBER_PTCS``-threshold (or ``MIN_NUMBER_STAR_PTCS``-threshold in case of ``cprofiles.calcGlobalShapesGx()``). If the global shape determination does not converge, it will appear as NaNs in the output.

*************************************
Velocity Dispersion Tensor Eigenaxes
*************************************

For Gadget-style HDF5 snapshot outputs one can calculate the velocity dispersion tensor eigenaxes by calling::

    cprofiles.calcGlobalVelShapesDM()

for global velocity shapes or ``cprofiles.calcLocalVelShapesDM()`` for local velocity shapes. In that case, additional output will be added to ``CAT_DEST``, reflecting the velocity-related morphological information:

* ``d_global_vdm_x.txt`` (``x`` being the snap string ``SNAP``) of shape (:math:`N_{\text{pass}}`,): ellipsoidal radii
* ``q_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,): q shape parameter
* ``s_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,): s shape parameter
* ``minor_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): minor axis
* ``inter_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): intermediate axis
* ``major_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): major axis
* ``cat_global_vdm_x.txt`` of length :math:`N_{\text{pass}}`: list of lists of indices of converged shape profiles, empty list entry [] if halo resolution is too low
* ``m_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,): masses of halos
* ``centers_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,3): centers of halos

The ``cprofiles.calcLocalVelShapesDM()`` command will dump files named ``d_local_vdm_x.txt`` etc.


