Riglog
=========================

- **Example**

.. code-block:: text

    # RIG VERSION: 0.3
    # RIG GIT COMMIT HASH: 08701fb
    # CODES: screen=0,imaging=1,position=2,lick=3,reward=4,lap=5,cam1=6,cam2=7,cam3=8,act0=21,act1=22
    # RIG CSV: code,time received,duino time,value
    # STARTED EXPERIMENT
    [2, 0, 7.0, 2508]
    [2, 0, 13.0, 2512]
    [2, 15, 19.0, 2516]
    [2, 15, 25.0, 2520]
    ...
    [6, 2711581, 2711520.0, 81427]
    [7, 2711581, 2711520.0, 81427]
    [2, 2711581, 2711525.0, 819]
    [8, 2711596, 2711529.0, 59873]
    [2, 2711596, 2711532.0, 823]
    # STOPPED EXPERIMENT



init riglog
---------------

.. code-block:: python

    from stimpyp import load_riglog

    root_path = ...  # riglog file path or riglog directory path
    riglog = load_riglog(root_path)



as array
---------------
Load as numpy array

.. code-block:: python

    print(riglog.dat)

- **output**

.. code-block:: text

    [[2.000000e+00 0.000000e+00 7.000000e+00 2.508000e+03]
     [2.000000e+00 0.000000e+00 1.300000e+01 2.512000e+03]
     [2.000000e+00 1.500000e+01 1.900000e+01 2.516000e+03]
     ...
     [2.000000e+00 2.711581e+06 2.711525e+06 8.190000e+02]
     [8.000000e+00 2.711596e+06 2.711529e+06 5.987300e+04]
     [2.000000e+00 2.711596e+06 2.711532e+06 8.230000e+02]]



log config
---------------
Get config dict for the log file

- **Refer to API**: :attr:`~stimpyp.base.RigConfig`

.. code-block:: python

    print(riglog.log_config)


- **output**

.. code-block:: text

    {'codes': {'act0': 21,
           'act1': 22,
           'cam1': 6,
           'cam2': 7,
           'cam3': 8,
           'imaging': 1,
           'lap': 5,
           'lick': 3,
           'position': 2,
           'reward': 4,
           'screen': 0},
     'commit_hash': '08701fb',
     'fields': ('code', 'time received', 'duino time', 'value'),
     'source_version': 'stimpy-bit',
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [1]
     'version': 0.3}


1. Based on field information, auto infer your stimpy version: {'stimpy-bit', 'stimpy-git', 'pyvstim'}

event time and value
------------------------

Get specific event information

- **Refer to API** :class:`~stimpyp.base.AbstractLog`


.. code-block:: python

    # get position event
    pos = riglog.position_event
    print(pos.value)  # value array
    print(pos.time)   # time array

    # get imaging (two-photon) event
    imaging = riglog.imaging_event
    print(imaging.value)
    print(imaging.time)

    # get camera event
    widefield_cam = riglog.camera_event['1P_cam']
    #                                  ^^^^^^^^^^ [1]
    print(widefield_cam.value)
    print(widefield_cam.time)
    print(widefield_cam.n_pulses)  # get number of imaging pulse

    # get 0-base lap index
    print(riglog.lap_event.value_index)


1. Use get item method with :attr:`~stimpyp.base.AbstractLog.CameraEvent`. options: ``facecam``, ``eyecam``, ``1P_cam``