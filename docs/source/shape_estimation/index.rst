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

.. math:: S_{ij} = \frac{1}{\sum_k m_k} \sum_k w_k m_k r_{k,i}r_{k,j}.

Here, :math:`m_k` is the mass of the :math:`k`-th particle, and :math:`r_{k} = (x_{k},y_{k},z_{k})^t` is the position vector with respect to the distribution's center (either mode or center of mass). The weight :math:`w_k` allows to define the most common shape tensors by choosing

* :math:`w_k = 1`, in which case each particle gets the same weight, or
* :math:`w_k = \frac{1}{r_k^2}` where :math:`r_k^2 = (x_{k})^2+(y_{k})^2+(z_{k})^2` is the distance squared of particle :math:`k` from the center of the cloud, or
* :math:`w_k = \frac{1}{r_{\text{ell},k}^2}` where :math:`r_{\text{ell},k}^2 = x_{\text{ell},k}^2+y_{\text{ell},k}^2+z_{\text{ell},k}^2` is the ellipsoidal radius, where :math:`(x_{\text{ell},k}, y_{\text{ell},k}, z_{\text{ell},k})` are the coordinates of particle :math:`k` in the eigenvector coordinate system of the ellipsoid. In other words, :math:`r_{\text{ell},k}` corresponds to the semi-major axis :math:`a` of the ellipsoid surface on which particle :math:`k` lies. The shape tensor with :math:`w_k = \frac{1}{r_{\text{ell},k}^2}` is also called the *reduced* shape tensor, a variant that penalizes particles at large radii.

Since the second weighting scheme with :math:`w_k = \frac{1}{r_k^2}` has recently fallen out of favour, see `Zemp et al. 2011 <https://arxiv.org/abs/1107.5582>`_, the other two schemes will be available by switching the boolean ``reduced``, see below.

To calculate shape profiles with *Cosmic Profiles*, let us assume we are dealing with

* a Gadget-style HDF5 snapshot output containing particle and halo/subhalo data in folders ``path/to/folder/snapdir_x`` and ``path/to/folder/groups_x`` with ``x`` typically a three-digit snapshot number identifier such as '042', respectively. Then we will first instantiate a ``DensShapeProfsHDF5`` object via::

    from cosmic_profiles import DensShapeProfsHDF5
    
    # Parameters
    L_BOX = np.float32(10) # cMpc/h
    HDF5_GROUP_DEST = "/path/to/groups_035"
    HDF5_SNAP_DEST = "/path/to/snapdir_035"
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
    WANT_RVIR = False # Whether or not we want quantities (e.g. D_LOGSTART) expressed with respect to the virial radius R_vir or overdensity radius R_200

    # Instantiate object
    cprofiles = DensShapeProfsHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP, SNAP_MAX, L_BOX, MIN_NUMBER_DM_PTCS, MIN_NUMBER_STAR_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER, WANT_RVIR)

with arguments explained in :ref:`the code reference<Cosmic Profiles Code Reference>`.

* a very general assortment of point clouds. There is no requirement on the nature of the point clouds whatsoever, yet the shape determination algorithm will perform better the closer the point clouds are to being truly ellipsoidal. Often, the process of identifying such point clouds in a simulation can be challenging, which is why we provide an :ref:`interface<AHF example>` showcasing how to use the 'Amiga Halo Finder' (AHF) via ``pynbody``. For now, we assume that we have identified the point clouds already and that ``idx_cat`` (list of lists) stores the indices of the particles belonging to the point clouds::
    
    from cosmic_profiles import DensShapeProfs
    
    # Parameters
    xyz = ... # application-dependent
    mass_array = ... # application-dependent
    idx_cat = ... # application-dependent
    r_vir = ... # application-dependent
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

    # Instantiate object
    cprofiles = DensShapeProfs(xyz, mass_array, idx_cat, r_vir, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, M_TOL, N_WALL, N_MIN, CENTER)

To retrieve the local (i.e. as a function of :math:`r_{\text{ell}}`) halo shape catalogue, we can invoke the command::

    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatLocal(reduced = False, shell_based = False).

The morphological information in ``d``, ``q``, ``s``, ``minor``, ``inter``, ``major``, ``obj_centers``, ``obj_masses`` represents the shape profiles. The arrays will contain NaNs whenever the shape determination did not converge. We consider the shape determination at a specific :math:`r_{\text{ell}}` to be converged if the fractional difference between consecutive eigenvalue fractions falls below ``M_TOL`` and the maximum number of iterations ``N_WALL`` is not yet achieved. The boolean ``reduced`` allows to select between the reduced shape tensor with weight :math:`w_k = \frac{1}{r_{\text{ell},k}^2}` and the regular shape tensor with :math:`w_k = 1`. The boolean ``shell_based`` allows to run the iterative shape identifier on ellipsoidal shells (= homoeoids) rather than ellipsoids. Note that ``shell_based = True`` should only be set if the number of particles resolving the objects is :math:`> \mathcal{O}(10^5)`. If :math:`N_{\text{pass}}` stands for the number of objects that are sufficiently resolved, then the 1D and 2D shape profile arrays will have the following format:

* ``d`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1): ellipsoidal radii
* ``q`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1): q shape parameter
* ``s`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1): s shape parameter
* ``minor`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1, 3): minor axes vs :math:`r_{\text{ell}}`
* ``inter`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1, 3): intermediate axes vs :math:`r_{\text{ell}}`
* ``major`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1, 3): major axes vs :math:`r_{\text{ell}}`
* ``obj_centers`` of shape (:math:`N_{\text{pass}}`,3): centers of objects 
* ``obj_masses`` of shape (:math:`N_{\text{pass}}`,): masses of objects.

.. note:: In the case of a ``DensShapeProfs`` object, there will be :math:`N_{\text{pass}}` objects with certain indices `obj` for which ``len(idx_cat[obj]) > MIN_NUMBER_PTCS``. In case of a ``DensShapeProfsHDF5`` object, the same holds true when identifying ``idx_cat`` with ``idx_cat = cprofiles.getIdxCat(obj_type = 'dm')`` and replacing ``MIN_NUMBER_PTCS`` by ``MIN_NUMBER_DM_PTCS``, and analogously for star particles in galaxies.

For post-processing purposes, one can dump the converged shape profiles in a destination ``CAT_DEST`` of choice via::
    
    cprofiles.dumpShapeCatLocal(CAT_DEST, reduced = False, shell_based = False),

where ``CAT_DEST`` is a string describing the absolute (or relative with respect to Python working diretory) path to the destination folder, e.g. '/path/to/cat'. The files added are

* ``d_local_x.txt`` (``x`` being the snap string ``SNAP``) of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1): ellipsoidal radii
* ``q_local_x.txt`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1): q shape parameter
* ``s_local_x.txt`` of shape (:math:`N_{\text{pass}}`, ``D_BINS`` + 1): s shape parameter
* ``minor_local_x.txt`` of shape (:math:`N_{\text{pass}}`, (``D_BINS`` + 1) * 3): minor axes vs :math:`r_{\text{ell}}`, have to apply ``minor_local_x.reshape(minor_local_x.shape[0], minor_local_x.shape[1]//3, 3)`` after loading with np.loadtxt()
* ``inter_local_x.txt`` of shape (:math:`N_{\text{pass}}`, (``D_BINS`` + 1) * 3): intermediate axes vs :math:`r_{\text{ell}}`, same here
* ``major_local_x.txt`` of shape (:math:`N_{\text{pass}}`, (``D_BINS`` + 1) * 3): major axes vs :math:`r_{\text{ell}}`, same here
* ``m_x.txt`` of shape (:math:`N_{\text{pass}}`,): masses of halos
* ``centers_x.txt`` of shape (:math:`N_{\text{pass}}`,3): centers of halos

.. note:: In case of a Gadget-style HDF5 snapshot output, specify ``cprofiles.getShapeCatLocal(reduced = False, shell_based = False, obj_type = 'dm')`` to calculate local halo (only the dark matter component of halos) shapes and ``cprofiles.getShapeCatLocal(reduced = False, shell_based = False, obj_type = 'gx')`` to calculate local galaxy shapes. The suffix of the output files when calling e.g. ``cprofiles.dumpShapeCatLocal(CAT_DEST, reduced = False, shell_based = False, obj_type = 'dm')`` will be modified accordingly to ``d_local_dm_x.txt``.

***************
Global Shapes
***************

Instead of shape profiles one might also be interested in obtaining the shape parameters and principal axes of the point clouds as a whole. This information can be obtained by calling::

    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatGlobal(reduced = False).

If a global shape calculations does not converge (which is rare), the corresponding entry in ``q`` etc. will feature a NaN. The index catalogue ``idx_cat = cprofiles.getIdxCat(obj_type)`` will have an empty entry at the corresponding location in the HDF5 case. In the generic point particle case, ``idx_cat = cprofiles.getIdxCat()`` will just return the ``idx_cat`` that is provided by the user, even if some entries have insufficient resolution. As with shape profiles, we can dump the global shape catalogue in a destination ``CAT_DEST`` of choice via::

    cprofiles.dumpShapeCatGlobal(CAT_DEST, reduced = False),

which will add the following files to the destination folder:

* ``d_global_x.txt`` (``x`` being the snap string ``SNAP``) of shape (:math:`N_{\text{pass}}`,): ellipsoidal radii
* ``q_global_x.txt`` of shape (:math:`N_{\text{pass}}`,): q shape parameter
* ``s_global_x.txt`` of shape (:math:`N_{\text{pass}}`,): s shape parameter
* ``minor_global_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): minor axis
* ``inter_global_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): intermediate axis
* ``major_global_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): major axis
* ``m_x.txt`` of shape (:math:`N_{\text{pass}}`,): masses of halos
* ``centers_x.txt`` of shape (:math:`N_{\text{pass}}`,3): centers of halos

In case of Gadget-style HDF5 files, invoke ``cprofiles.getShapeCatGlobal(reduced = False, obj_type = 'dm')`` to calculate global halo shapes and ``cprofiles.getShapeCatGlobal(reduced = False, obj_type = 'gx')`` to calculate global galaxy shapes. To dump the files, call ``cprofiles.dumpShapeCatGlobal(CAT_DEST, reduced = False, obj_type = 'dm')`` or ``cprofiles.dumpShapeCatGlobal(CAT_DEST, reduced = False, obj_type = 'gx')``.

.. note:: As previously, :math:`N_{\text{pass}}` denotes the number of halos that pass the ``MIN_NUMBER_PTCS``-threshold (or ``MIN_NUMBER_DM_PTCS``-threshold in case of ``cprofiles.getShapeCatGlobal(reduced = False, obj_type = 'dm')`` or ``MIN_NUMBER_STAR_PTCS``-threshold in case of ``cprofiles.getShapeCatGlobal(reduced = False, obj_type = 'gx')``). If the global shape determination for a sufficiently resolved object does not converge, it will appear as NaNs in the output.

*************************************
Velocity Dispersion Tensor Eigenaxes
*************************************

For Gadget-style HDF5 snapshot outputs one can calculate the velocity dispersion tensor eigenaxes by calling::

    d, q, s, minor, inter, major, obj_centers, obj_masses = cprofiles.getShapeCatVelLocal(reduced = False, shell_based = False, obj_type = 'dm')

for local velocity shapes or ``cprofiles.getShapeCatVelGlobal(reduced = False, obj_type = 'dm')`` for global velocity shapes. When calling e.g. ``cprofiles.dumpShapeCatVelGlobal(CAT_DEST, reduced = False, obj_type = 'dm')``, the overall halo velocity dispersion tensor shapes of the following format will be added to ``CAT_DEST``:

* ``d_global_vdm_x.txt`` (``x`` being the snap string ``SNAP``) of shape (:math:`N_{\text{pass}}`,): ellipsoidal radii
* ``q_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,): q shape parameter
* ``s_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,): s shape parameter
* ``minor_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): minor axis
* ``inter_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): intermediate axis
* ``major_global_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`, 3): major axis
* ``m_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,): masses of halos
* ``centers_vdm_x.txt`` of shape (:math:`N_{\text{pass}}`,3): centers of halos

The ``cprofiles.dumpShapeCatVelGlobal(CAT_DEST, reduced = False, obj_type = 'dm')`` command will dump files named ``d_local_vdm_x.txt`` etc.


