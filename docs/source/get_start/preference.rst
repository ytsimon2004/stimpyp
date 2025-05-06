Preference
===========
Experimental Preference(config) file as Python ``TypedDict``

Refer to :class:`~stimpyp.preference.PreferenceDict`


.. code-block:: python

    class PreferenceDict(TypedDict, total=False):
        user: str
        userPrefix: str
        expname: str  # git only
        defaultExperimentType: str
        default_imaging_mode: str

        logFolder: PathLike
        protocolsFolder: PathLike
        controllerFolder: PathLike
        stimsFolder: PathLike
        tmpFolder: PathLike  # git only

        monitor: list[MonitorDict]
        use_monitor: Union[int, list[int]]

        # PanoDisplay
        vr_flag: bool
        vrFolder: PathLike

        # PyGameVRDisplay
        textureFolder: PathLike
        raycaster: Literal['default', 'numpy', 'numba']

        # Photo indicator
        flashIndicator: bool
        flashIndicatorMode: int
        flashIndicatorParameters: FlashIndicatorParameters

        # Network
        labcams: NetworkControllerDict
        scanbox: NetworkControllerDict
        pycams: NetworkControllerDict
        spikeglx: NetworkControllerDict

        rig: RigDict
        warp: MinotorWarpDict

        # runtime append
        _controllers_data_folder: Path
        _controllers_protocol_folder: str
        _controllers_folder_read_flag: bool


init preference
------------------

**Infer from riglog**

.. code-block:: python

    from stimpyp import load_riglog

    file = ...  # riglog file path or riglog directory path
    riglog = load_riglog(file)
    pref = riglog.get_preferences()


**Load file**

.. code-block:: python

    from stimpyp import load_preference
    file = ...  # preference file path
    protocol = load_preference(file)