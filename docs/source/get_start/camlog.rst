Camlog
=========

Camera log parsing

- **Refer to API**: :doc:`../api/stimpyp.camlog`


labcams versus pycams
-----------------------

- **Example of labcams log**

.. code-block:: text

    # Camera: facecam log file
    # Date: 15-03-2021
    # labcams version: 0.2
    # Log header:frame_id,timestamp
    # [21-03-15 18:13:09] - I:\data\facecam\210315_YW006__2P_YW\run00_181302_ori_sqr_12dir_2tf_3sf_bas\20210315_run000_00000000.tif
    # [21-03-15 18:13:09] - Queue: 40
    1,0.0014252
    2,0.0347656
    3,0.0680934
    4,0.1014338
    5,0.1347617
    6,0.1680901
    ...
    254,8.4346838
    255,8.4680117
    256,8.5013401
    # [21-03-15 18:13:16] - I:\data\facecam\210315_YW006__2P_YW\run00_181302_ori_sqr_12dir_2tf_3sf_bas\20210315_run000_00000001.tif
    # [21-03-15 18:13:16] - Queue: 0
    ...


- **Example of pycams log**

.. code-block:: text

    # Commit hash: 50082af
    # Log header: frame_id,timestamp
    1,0.048
    2,0.142
    3,0.211
    4,0.287
    5,0.347
    6,0.434
    7,0.507
    8,0.586
    9,0.665
    10,0.732
    11,0.808
    12,0.887
    13,0.966
    14,1.048
    15,1.125
    16,1.204
    17,1.283
    18,1.349
    19,1.426
    20,1.512
    21,1.585
    22,1.665
    23,1.745
    24,1.824
    ...


as dataframe
---------------

.. code-block:: python

    from stimpyp import read_camlog

    file = .... # .log or .camlog file path
    camera_version = ... # either labcams or pycams
    camlog = read_camlog(file, camera_version=camera_version)
    print(camlog.to_polars())


.. code-block:: text

    ┌──────────┬──────────┐
    │ frame_id ┆ time     │
    │ ---      ┆ ---      │
    │ i64      ┆ f64      │
    ╞══════════╪══════════╡
    │ 1        ┆ 0.048    │
    │ 2        ┆ 0.142    │
    │ 3        ┆ 0.211    │
    │ 4        ┆ 0.287    │
    │ 5        ┆ 0.347    │
    │ …        ┆ …        │
    │ 14657    ┆ 1123.081 │
    │ 14658    ┆ 1123.145 │
    │ 14659    ┆ 1123.224 │
    │ 14660    ┆ 1123.309 │
    │ 14661    ┆ 1123.367 │
    └──────────┴──────────┘