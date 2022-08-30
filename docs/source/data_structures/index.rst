.. _Data Structures:

Supported Data Structures
==========================

To estimate and fit shape and / or density profiles with *Cosmic Profiles*, at the moment we support the following data structures.

* *GADGET*-style HDF5 snapshots containing particle and halo/subhalo data in folders ``path/to/folder/snapdir_x`` and ``path/to/folder/groups_x`` with ``x`` typically a three-digit snapshot number identifier such as '042', respectively. Note that the HDF5 snapshot of the simulation must have been written with FoF / SUBFIND turned on in the simulation (e.g. *Arepo* or *GADGET-4*), otherwise *Cosmic Profiles* will not know how to identify the central subhalos of dark matter, gas or star particles. To calculate shape profiles, we instantiate a ``DensShapeProfsHDF5`` object via::

    from cosmic_profiles import DensShapeProfsHDF5, updateInUnitSystem, updateOutUnitSystem
    
    # Parameters
    updateInUnitSystem(in_unit_length_in_cm = 3.085678e21, in_unit_mass_in_g = 1.989e43, in_unit_velocity_in_cm_per_s = 1e5)
    updateOutUnitSystem(out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)
    L_BOX = np.float32(10000) # kpc/h
    HDF5_GROUP_DEST = "/path/to/groups_035"
    HDF5_SNAP_DEST = "/path/to/snapdir_035"
    D_LOGSTART = -2
    D_LOGEND = 1
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    IT_TOL = np.float32(1e-2)
    IT_WALL = 100
    IT_MIN = 10
    SNAP = '035'
    CENTER = 'mode'
    MIN_NUMBER_PTCS = 200
    RVIR_OR_R200 = 'R200'
    OBJ_TYPE = 'dm'

    # Instantiate object
    cprofiles = DensShapeProfsHDF5(HDF5_SNAP_DEST, HDF5_GROUP_DEST, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, CENTER, RVIR_OR_R200, OBJ_TYPE)

with arguments and public methods explained in detail in :ref:`the code reference<Cosmic Profiles Code Reference>` but summarized here for completeness.

.. dropdown:: User Parameters for DensShapeProfsHDF5

  * ``HDF5_GROUP_DEST`` and ``HDF5_SNAP_DEST``: path to simulation snapshot
  * ``SNAP``: snapshot identifier, used when dumping files etc.
  * ``L_BOX``: simulation box side length (i.e. periodicity of box) in units of ``config.InUnitLength_in_cm``
  * ``MIN_NUMBER_PTCS``: minimum number of particles for objects to qualify for analyses (e.g. shape analysis)
  * ``D_LOGSTART`` and ``D_LOGEND``: logarithm of minimum and maximum ellipsoidal radius of interest, in units of R200 or Rvir (depending on ``RVIR_OR_R200``) of parent halo
  * ``D_BINS``: number of bins to consider for shape profiling 
  * ``IT_TOL``: convergence tolerance in shape estimation algorithm, eigenvalue fractions must differ by less than ``IT_TOL`` for algorithm to halt
  * ``IT_WALL``: maximum permissible number of iterations in shape estimation algorithm
  * ``IT_MIN``: minimum number of particles (DM, gas or star particles depending on ``OBJ_TYPE``) in any iteration, if undercut, shape is unclassified
  * ``CENTER``: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density) or 'com' (center of mass) of each object (= DM halo, gas halo or star particle halo)
  * ``RVIR_OR_R200``: 'Rvir' if we want quantities (e.g. D_LOGSTART) to be expressed with respect to the virial radius R_vir, 'R200' for the overdensity radius R_200
  * ``OBJ_TYPE``: which simulation particles to consider, 'dm', 'gas' or 'stars'

.. dropdown:: Public Methods of DensShapeProfsHDF5

  In the following, 'object' refers to either a DM halo, a gas halo or a star particle halo, depending on ``OBJ_TYPE``. The generic public methods are

  * ``getPartType()``: return particle type number in simulation, ``0`` for ``OBJ_TYPE=gas``, ``1`` for ``OBJ_TYPE=dm`` and ``4`` for ``OBJ_TYPE=stars``
  * ``getXYZMasses()``: retrieve positions in units of ``config.OutUnitLength_in_cm`` and masses of particles in units of ``config.OutUnitMass_in_g``
  * ``getVelXYZMasses()``: retrieve velocities of particles in units of ``config.OutUnitVelocity_in_cm_per_s``
  * ``getR200()``: fetch R200 value of all objects (that have sufficient resolution as is implicitly assumed everywhere) in units of ``config.OutUnitLength_in_cm``
  * ``getIdxCat()``: fetch index catalogue (each row contains indices of particles belonging to an object) and object sizes (number of particles in each object)
  * ``getMassesCenters(select)``: calculate and return centers (in units of ``config.OutUnitLength_in_cm``) and total masses of objects (in units of ``config.OutUnitMass_in_g``)
  * ``getObjInfo()``: print basic info such as number of objects with sufficient resolution, number of subhalos, number of objects (halos) that have no subhalos etc.,

  the density profiling-related public methods are
  
  * ``estDensProfs(ROverR200, select, direct_binning = True, spherical = True, reduced = False, shell_based = False)``: estimate density profiles at normalized radii ``ROverR200``
  * ``fitDensProfs(dens_profs, ROverR200, method, select)``: get best-fit results for density profile fitting
  * ``estConcentrations(dens_profs, ROverR200, method, select)``: get best-fit concentration values from density profile fitting
  * ``plotDensProfs(dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, method, nb_bins, VIZ_DEST, select)``: draw some simplistic density profiles and save in ``VIZ_DEST``

  while the shape profiling-related public methods are
  
  * ``getShapeCatLocal(select, reduced = False, shell_based = False)``: estimate and return shape profiles  
  * ``getShapeCatGlobal(select, reduced = False)``: estimate and return global shape data
  * ``vizLocalShapes(obj_numbers, VIZ_DEST, reduced = False, shell_based = False)``: visualize shape profiles of objects with numbers ``obj_numbers`` and save in ``VIZ_DEST``
  * ``vizGlobalShapes(obj_numbers, VIZ_DEST, reduced = False)``: visualize global shapes of objects with numbers ``obj_numbers`` and save in ``VIZ_DEST``
  * ``plotGlobalEpsHist(HIST_NB_BINS, VIZ_DEST, select)``: plot histogram of overall (= global) ellipticities (complex magnitude)
  * ``plotLocalEpsHist(frac_r200, HIST_NB_BINS, VIZ_DEST, select)``: plot histogram of local ellipticities (complex magnitude) at depth ``frac_r200``
  * ``plotLocalTHist(HIST_NB_BINS, VIZ_DEST, frac_r200, select, reduced = False, shell_based = False)``: plot histogram of local triaxiality at depth ``frac_r200``
  * ``plotGlobalTHist(HIST_NB_BINS, VIZ_DEST, select, reduced = False)``: plot histogram of global triaxiality
  * ``plotShapeProfs(nb_bins, VIZ_DEST, select, reduced = False, shell_based = False)``: plot shape profiles, also mass bin-decomposed ones
  * ``dumpShapeCatLocal(CAT_DEST, select, reduced = False, shell_based = False)``: dumps all relevant local shape data into ``CAT_DEST``
  * ``dumpShapeCatGlobal(CAT_DEST, select, reduced = False, shell_based = False)``: dumps all relevant global shape data into ``CAT_DEST``.

* very general assortments of point clouds. There is no requirement on the nature of the point clouds whatsoever, yet the shape determination algorithm will perform better the closer the point clouds are to being truly ellipsoidal. Often, the process of identifying such point clouds in a simulation can be challenging, which is why we provide an :ref:`interface<AHF example>` showcasing how to use the 'Amiga Halo Finder' (AHF) via ``pynbody``. For now, we assume that we have identified the point clouds already and that ``idx_cat`` (list of lists) stores the indices of the particles belonging to the point clouds::
    
    from cosmic_profiles import DensShapeProfs, updateInUnitSystem, updateOutUnitSystem
    
    # Parameters
    updateInUnitSystem(in_unit_length_in_cm = 3.085678e24, in_unit_mass_in_g = 1.989e33, in_unit_velocity_in_cm_per_s = 1e5)
    updateOutUnitSystem(out_unit_length_in_cm = 3.085678e24, out_unit_mass_in_g = 1.989e33, out_unit_velocity_in_cm_per_s = 1e5)
    xyz = ... # application-dependent
    mass_array = ... # application-dependent
    idx_cat = ... # application-dependent
    r_vir = ... # application-dependent
    SNAP = '035'
    L_BOX = np.float32(10) # cMpc/h
    D_LOGSTART = -2
    D_LOGEND = 1
    D_BINS = 30 # If D_LOGSTART == -2 D_LOGEND == 1, 60 corresponds to shell width of 0.05 dex
    IT_TOL = np.float32(1e-2)
    IT_WALL = 100
    IT_MIN = 10
    SNAP = '035'
    CENTER = 'mode'
    MIN_NUMBER_PTCS = 200

    # Instantiate object
    cprofiles = DensShapeProfs(xyz, mass_array, idx_cat, r_vir, SNAP, L_BOX, MIN_NUMBER_PTCS, D_LOGSTART, D_LOGEND, D_BINS, IT_TOL, IT_WALL, IT_MIN, CENTER)

.. dropdown:: User Parameters for DensShapeProfs

  * ``xyz``: positions of all (simulation) particles in units of ``config.InUnitLength_in_cm``
  * ``mass_array``: masses of all (simulation) particles in units of ``config.InUnitMass_in_g``
  * ``idx_cat``: each entry of the list is a list containing indices (to ``xyz`` and ``mass_array``, respectively) of particles belonging to an object
  * ``r_vir``: virial radii of the parent halos in units of ``config.InUnitLength_in_cm``
  * ``SNAP``: snapshot identifier, used when dumping files etc.
  * ``L_BOX``: simulation box side length (i.e. periodicity of box) in units of ``config.InUnitLength_in_cm``
  * ``MIN_NUMBER_PTCS``: minimum number of particles for objects to qualify for analyses (e.g. shape analysis)
  * ``D_LOGSTART`` and ``D_LOGEND``: logarithm of minimum and maximum ellipsoidal radius of interest, in units of R200 or Rvir (depending on ``RVIR_OR_R200``) of parent halo
  * ``D_BINS``: number of bins to consider for shape profiling 
  * ``IT_TOL``: convergence tolerance in shape estimation algorithm, eigenvalue fractions must differ by less than ``IT_TOL`` for algorithm to halt
  * ``IT_WALL``: maximum permissible number of iterations in shape estimation algorithm
  * ``IT_MIN``: minimum number of particles (DM, gas or star particles depending on ``OBJ_TYPE``) in any iteration, if undercut, shape is unclassified
  * ``CENTER``: shape quantities will be calculated with respect to CENTER = 'mode' (point of highest density) or 'com' (center of mass) of each object

.. dropdown:: Public Methods of DensShapeProfs

  In the following, 'object' refers to the objects that are defined via the indices ``idx_cat`` provided by the user. The generic public methods are
  
  * ``getXYZMasses()``: retrieve positions in units of ``config.OutUnitLength_in_cm`` and masses of particles in units of ``config.OutUnitMass_in_g``
  * ``getR200()``: fetch R200 value of all objects (that have sufficient resolution as is implicitly assumed everywhere) in units of ``config.OutUnitLength_in_cm``
  * ``getIdxCat()``: fetch index catalogue (each row contains indices of particles belonging to an object) and object sizes (number of particles in each object)
  * ``getMassesCenters(select)``: calculate and return centers (in units of ``config.OutUnitLength_in_cm``) and total masses of objects (in units of ``config.OutUnitMass_in_g``)
  * ``getObjInfo()``: print basic info such as number of objects with sufficient resolution etc.,

  the density profiling-related public methods are
  
  * ``estDensProfs(ROverR200, select, direct_binning = True, spherical = True)``: estimate density profiles at normalized radii ``ROverR200``
  * ``fitDensProfs(dens_profs, ROverR200, method, select)``: get best-fit results for density profile fitting
  * ``estConcentrations(dens_profs, ROverR200, method, select)``: get best-fit concentration values from density profile fitting
  * ``plotDensProfs(dens_profs, ROverR200, dens_profs_fit, ROverR200_fit, method, nb_bins, VIZ_DEST, select)``: draw some simplistic density profiles and save in ``VIZ_DEST``
  
  while the shape profiling-related public methods are
  
  * ``getShapeCatLocal(select, reduced = False, shell_based = False)``: estimate and return shape profiles  
  * ``getShapeCatGlobal(select, reduced = False)``: estimate and return global shape data
  * ``vizLocalShapes(obj_numbers, VIZ_DEST, reduced = False, shell_based = False)``: visualize shape profiles of objects with numbers ``obj_numbers`` and save in ``VIZ_DEST``
  * ``vizGlobalShapes(obj_numbers, VIZ_DEST, reduced = False)``: visualize global shapes of objects with numbers ``obj_numbers`` and save in ``VIZ_DEST``
  * ``plotGlobalEpsHist(HIST_NB_BINS, VIZ_DEST, select)``: plot histogram of overall (= global) ellipticities (complex magnitude)
  * ``plotLocalEpsHist(frac_r200, HIST_NB_BINS, VIZ_DEST, select)``: plot histogram of local ellipticities (complex magnitude) at depth ``frac_r200``
  * ``plotLocalTHist(HIST_NB_BINS, VIZ_DEST, frac_r200, select, reduced = False, shell_based = False)``: plot histogram of local triaxiality at depth ``frac_r200``
  * ``plotGlobalTHist(HIST_NB_BINS, VIZ_DEST, select, reduced = False)``: plot histogram of global triaxiality
  * ``plotShapeProfs(nb_bins, VIZ_DEST, select, reduced = False, shell_based = False)``: plot shape profiles, also mass bin-decomposed ones
  * ``dumpShapeCatLocal(CAT_DEST, select, reduced = False, shell_based = False)``: dumps all relevant local shape data into ``CAT_DEST``
  * ``dumpShapeCatGlobal(CAT_DEST, select, reduced = False, shell_based = False)``: dumps all relevant global shape data into ``CAT_DEST``.

