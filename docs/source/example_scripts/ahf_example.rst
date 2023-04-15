.. _AHF example:

AHF Example
============

|pic1|

.. |pic1| image:: RhoHalo.png
   :width: 60%

.. note:: Additional packages to install: `pynbody <https://pynbody.github.io/pynbody/>`_ and `nbodykit <https://nbodykit.readthedocs.io/en/latest/>`_.

We mentioned that identifying halos and galaxies for a simulation box is a challenging task. Most state-of-the-art halo finders have no direct Python wrappers, yet `pynbody <https://pynbody.github.io/pynbody/>`_ allows to overcome this limitation by wrapping to **most** major halo finders, including the Amiga Halo Finder (AHF), Rockstar and SubFind.

For this reason we demonstrate in this example how one can use the ``s.halos()`` command of ``pynbody`` to identify halos in a simulation box. For ``pynbody``'s AHF-invoking ``s.halos()`` command to work with a Gadget output, one needs to install AHF and place the executable into ``~/bin`` (or extend the ``$PATH`` variable)

.. code-block:: bash

    $ wget popia.ft.uam.es/AHF/files/ahf-v1.0-111.tgz
    $ tar zxf ahf-v1.0-111.tgz; cd ahf-v1.0-111   # extract
    $ make AHF                                    # build
    $ cp bin/./AHF-v1.0-111 ~/bin/                # copy into $PATH
    $ echo 'PATH=$PATH:~/bin' >> ~/.bashrc        # make $PATH aware of AHF

Then, modify ``_run_ahf(self, sim)`` in ``/path/to/pynbody/halo/ahf.py`` to::

    def _run_ahf(self, sim):
        typecode = '60' # '61'
        import pynbody.units as units
        # find AHFstep

        groupfinder = config_parser.get('AHFCatalogue', 'Path')

        if groupfinder == 'None':
            for directory in os.environ["PATH"].split(os.pathsep):
                ahfs = glob.glob(os.path.join(directory, "AHF*"))
                for iahf, ahf in enumerate(ahfs):
                    # if there are more AHF*'s than 1, it's not the last one, and
                    # it's AHFstep, then continue, otherwise it's OK.
                    if ((len(ahfs) > 1) & (iahf != len(ahfs) - 1) &
                            (os.path.basename(ahf) == 'AHFstep')):
                        continue
                    else:
                        groupfinder = ahf
                        break

        if not os.path.exists(groupfinder):
            raise RuntimeError("Path to AHF (%s) is invalid" % groupfinder)

        if (os.path.basename(groupfinder) == 'AHFstep'):
            isAHFstep = True
        else:
            isAHFstep = False
        # build units file
        if isAHFstep:
            f = open('tipsy.info', 'w')
            f.write(str(sim.properties['omegaM0']) + "\n")
            f.write(str(sim.properties['omegaL0']) + "\n")
            f.write(str(sim['pos'].units.ratio(
                units.kpc, a=1) / 1000.0 * sim.properties['h']) + "\n")
            f.write(
                str(sim['vel'].units.ratio(units.km / units.s, a=1)) + "\n")
            f.write(str(sim['mass'].units.ratio(units.Msol)) + "\n")
            f.close()
            # make input file
            f = open('AHF.in', 'w')
            f.write(sim._filename + " " + str(typecode) + " 1\n")
            f.write(sim._filename + "\n256\n5\n5\n0\n0\n0\n0\n")
            f.close()
        else:
            # make input file
            f = open('AHF.in', 'w')

            lgmax = np.min([int(2 ** np.floor(np.log2(
                1.0 / np.min(sim['eps'])))), 131072])
            #lgmax = np.min([int(2 ** np.floor(np.log2(
            #    1.0 / 0.19))), 131072])
            # hardcoded maximum 131072 might not be necessary

            print(config_parser.get('AHFCatalogue', 'Config', vars={
                'filename': str(sim._filename),
                'typecode': int(typecode),
                'gridmax': int(lgmax)
            }), file=f)

            print(config_parser.get('AHFCatalogue', 'ConfigGadget', vars={
                'omega0': sim.properties['omegaM0'],
                'lambda0': sim.properties['omegaL0'],
                'boxsize': sim['pos'].units.ratio('Mpc a h^-1', **sim.conversion_context()),
                'vunit': sim['vel'].units.ratio('km s^-1 a', **sim.conversion_context()),
                'munit': sim['mass'].units.ratio('Msol h^-1', **sim.conversion_context()),
                'eunit': 0.03  # surely this can't be right?
            }), file=f)

            f.close()

        if (not os.path.exists(sim._filename)):
            os.system("gunzip " + sim._filename + ".gz")
        # determine parallel possibilities

        if os.path.exists(groupfinder):
            # run it
            os.system(groupfinder + " AHF.in")
            return

and the [AHFCatalogue] section in ``/path/to/pynbody/config.ini`` to::

    [AHFCatalogue]
    # settings for the AHF Catalogue reader

    AutoRun: True
    # automatically attempt to run AHF if no catalogue can be found
    # on disk

    Path: None
    # /path/to/AHF, or None to attempt to find it in your $PATH

    AutoGrp: False
    # set to true to automatically create a 'grp' array on load
    # The grp array

    AutoPid: False
    # set to true to automatically create a 'pid' array on load
    # the PID array is another way to get the particle IDs in the ancestor snapshot,
    # but the framework provides h[n].get_index_list(f) for halo catalogue h and
    # base snapshot f, so you probably don't need AutoPid

    Config:   [AHF]
              ic_filename = %(filename)s
              ic_filetype = %(typecode)s
              outfile_prefix = %(filename)s
              LgridDomain = 128
              LgridMax = %(gridmax)s
              NperDomCell = 5
              NperRefCell = 5
              VescTune = 1.5
              NminPerHalo = 50
              RhoVir = 0
              Dvir = 200
              MaxGatherRad = 10.0

    ConfigGadget:     [GADGET]
              GADGET_MUNIT = 1.0e10
              GADGET_LUNIT = 1.0e-3

In this example, we generate a mock universe using ``nbodykit``, save the universe to a Gadget 2 file, load the Gadget 2 file with ``pynbody``, identify halos with AHF and estimate shape profiles with CosmicProfiles.

If ``pynbody.plot.image(halos[2].d, width = '500 kpc', cmap=plt.cm.Greys, units = 'Msol kpc^-2')`` fails, modify the argument ``cen_size`` in the ``center()`` function of ``/path/to/pynbody/analysis/halo.py`` to something like ``cen_size="10 kpc"``.

.. literalinclude :: ../../../example_scripts/apply_ahf.py
   :language: python
