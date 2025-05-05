Stimlog
==========
Visual stimulation log file

- **Example from Bitbucket version**

.. code-block:: text

    # VLOG HEADER:code, presentTime, iStim, iTrial, photo, contrast, ori, sf, phase, stim_idx
    # CODES: vstim=10
    # Commit: 08701fb
    # Vstim trigger on: imaging,1.0
    10,0.006455,None,0,1
    10,0.278225,None,0,1
    10,0.284406,None,0,1
    10,0.301736,None,0,1
    10,0.305435,None,0,1
    10,0.318425,None,0,1
    10,0.319513,None,0,1
    10,0.335153,None,0,1
    10,0.338386,None,0,1
    ...
    10,902.034956,None,0,1
    # Started state machine v1.2 - timing sync to rig
    # CODES: stateMachine=20
    # STATE HEADER: code,elapsed,cycle,newState,oldState,stateElapsed,trialType
    20,902633,0,1,0,902633,0
    10,902.068384,16,0,1
    10,902.085096,16,0,1
    ...
    10,2710.922595,10,5,1
    10,2710.939292,10,5,1
    10,2710.956034,10,5,1
    10,2710.972765,10,5,1
    10,2710.989471,10,5,1
    # END OF VSTIM
    # Missed 11 frames


- **Example from Github version**


.. code-block:: text

    #### LOG NAME: stimpy_main_logger
    #### CODE VERSION: commit hash: 88c4705 - tags: ['']
    #### Format: source_id time source_infos
    # Rig trigger on: imaging,0.0
    ### START 27.971045
    ## 0:Gratings ['duration', 'contrast', 'ori', 'phase', 'pos', 'size', 'flick', 'interpolate', 'mask', 'sf', 'tf', 'opto', 'pattern']
    ## 1:PhotoIndicator ['state', 'size', 'pos', 'units', 'mode', 'frames', 'enabled']
    ## 2:StateMachine ['state', 'prev_state']
    ## 3:LogDict ['block_nr', 'trial_nr', 'condition_nr', 'trial_type']
    0 929.89731 [3, 1, 120, 0, [0, 0], [200, 200], 0, True, None, 0.04, 4, 0, 'sqr']
    1 929.89731 [False, 35, [740, 370], 'pix', 0, 20, True]
    2 929.89731 [<States.SHOW_BLANK: 1>, <States.STIM_SELECT: 0>]
    3 929.89731 [0, 0, 35, 1]
    2 931.907313 [<States.SHOW_STIM: 2>, <States.SHOW_BLANK: 1>]
    0 931.917313 [3, 1, 120, 0.06666666666666667, [0, 0], [200, 200], 0, True, None, 0.04, 4, 0, 'sqr']
    1 931.917313 [True, 35, [740, 370], 'pix', 0, 20, True]
    0 931.947313 [3, 1, 120, 0.13333333333333333, [0, 0], [200, 200], 0, True, None, 0.04, 4, 0, 'sqr']
    0 931.957313 [3, 1, 120, 0.2, [0, 0], [200, 200], 0, True, None, 0.04, 4, 0, 'sqr']
    ...
    0 2733.017838 [3, 1, 180, 2.8666666666666667, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.037838 [3, 1, 180, 2.8833333333333333, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.057838 [3, 1, 180, 2.9, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.077838 [3, 1, 180, 2.9166666666666665, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.087838 [3, 1, 180, 2.933333333333333, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.107838 [3, 1, 180, 2.95, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.117838 [3, 1, 180, 2.966666666666667, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    0 2733.137838 [3, 1, 180, 2.9833333333333334, [0, 0], [200, 200], 0, True, None, 0.16, 1, 0, 'sqr']
    2 2733.137838 [<States.STIM_SELECT: 0>, <States.SHOW_STIM: 2>]
    ### END 3635.080104
    # 0 removed
    # 1 removed
    # 2 removed
    # 3 removed
    # Missed 247 frames
    #### LOG NAME: stimpy_main_logger
    #### CODE VERSION: commit hash: 88c4705 - tags: ['']
    #### Format: source_id time source_infos


- **Example from Github version (the most recent update)**

.. code-block:: text

    TODO add



init stimlog
--------------------

- **Infer from riglog**

.. code-block:: python

    from stimpyp.parser import load_riglog

    file = ...  # riglog file path or riglog directory path
    riglog = load_riglog(file, diode_offset=True)
    #                          ^^^^^^^^^^^^^^^^^ [1]
    stimlog = riglog.get_stimlog()
    #                ^^^^^^^^^^^^ [2]

1. Do the diode offset to sync the time between riglog and stimlog, then both logs shared the same timescale
2. Get the corresponding stimlog object. either :attr:`~stimpyp.parser.stimpy_core.Stimlog` (bitbucket version parser)
or :attr:`~stimpyp.parser.stimpy_git.StimlogGit` (github version parser)


Common attributes
---------------------
Common usage across different stimpy version

- **Refer to API**: :attr:`~stimpyp.parser.base.AbstractStimlog`


config information
^^^^^^^^^^^^^^^^^^^^^^^^^
Get stimlog config information

.. code-block:: python

    # commit hash, missing frames ...
    print(stimlog.config)

    # bitbucket: {10: 'vstim', 20: 'stateMachine'}
    # github: {0: 'Gratings', 1: 'PhotoIndicator', 2: 'StateMachine', 3: 'LogDict'}...
    print(stimlog.log_info)

    # bitbucket: {0: ['duration', 'contrast', 'ori', 'phase', 'pos', 'size', 'flick', 'interpolate', 'mask', 'sf', 'tf', 'opto', 'pattern'], ...}
    # github: {0: ['duration', 'contrast', 'ori', 'phase', 'pos', 'size', 'flick', 'interpolate', 'mask', 'sf', 'tf', 'opto', 'pattern'], ...}
    print(stimlog.log_header)



visual stim dataframe
^^^^^^^^^^^^^^^^^^^^^^^^^
Get visual stimulation dataframe

.. code-block:: python

    print(stimlog.get_visual_stim_dataframe())

- **output of bitbucket version**

.. code-block:: text

    ┌─────────────┬───────┬────────┬───────┬───┬───────┬──────┬──────────┬──────────┐
    │ presentTime ┆ iStim ┆ iTrial ┆ photo ┆ … ┆ ori   ┆ sf   ┆ phase    ┆ stim_idx │
    │ ---         ┆ ---   ┆ ---    ┆ ---   ┆   ┆ ---   ┆ ---  ┆ ---      ┆ ---      │
    │ f64         ┆ f64   ┆ f64    ┆ f64   ┆   ┆ f64   ┆ f64  ┆ f64      ┆ f64      │
    ╞═════════════╪═══════╪════════╪═══════╪═══╪═══════╪══════╪══════════╪══════════╡
    │ 904.074325  ┆ 16.0  ┆ 0.0    ┆ 0.0   ┆ … ┆ 120.0 ┆ 0.08 ┆ 0.016667 ┆ 1.0      │
    │ 904.091047  ┆ 16.0  ┆ 0.0    ┆ 0.0   ┆ … ┆ 120.0 ┆ 0.08 ┆ 0.033333 ┆ 2.0      │
    │ 904.107793  ┆ 16.0  ┆ 0.0    ┆ 0.0   ┆ … ┆ 120.0 ┆ 0.08 ┆ 0.05     ┆ 3.0      │
    │ 904.124567  ┆ 16.0  ┆ 0.0    ┆ 0.0   ┆ … ┆ 120.0 ┆ 0.08 ┆ 0.066667 ┆ 4.0      │
    │ 904.141202  ┆ 16.0  ┆ 0.0    ┆ 0.0   ┆ … ┆ 120.0 ┆ 0.08 ┆ 0.083333 ┆ 5.0      │
    │ …           ┆ …     ┆ …      ┆ …     ┆ … ┆ …     ┆ …    ┆ …        ┆ …        │
    │ 1806.201863 ┆ 10.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 300.0 ┆ 0.04 ┆ 2.933333 ┆ 176.0    │
    │ 1806.218558 ┆ 10.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 300.0 ┆ 0.04 ┆ 2.95     ┆ 177.0    │
    │ 1806.235269 ┆ 10.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 300.0 ┆ 0.04 ┆ 2.966667 ┆ 178.0    │
    │ 1806.251987 ┆ 10.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 300.0 ┆ 0.04 ┆ 2.983333 ┆ 179.0    │
    │ 1806.268704 ┆ 10.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 300.0 ┆ 0.04 ┆ 2.983333 ┆ 179.0    │
    └─────────────┴───────┴────────┴───────┴───┴───────┴──────┴──────────┴──────────┘


- **output of github version**

.. code-block:: text

    ┌─────────────┬──────────┬──────────┬─────┬───┬──────┬─────┬──────┬─────────┐
    │ time        ┆ duration ┆ contrast ┆ ori ┆ … ┆ sf   ┆ tf  ┆ opto ┆ pattern │
    │ ---         ┆ ---      ┆ ---      ┆ --- ┆   ┆ ---  ┆ --- ┆ ---  ┆ ---     │
    │ f64         ┆ i64      ┆ i64      ┆ i64 ┆   ┆ f64  ┆ i64 ┆ i64  ┆ str     │
    ╞═════════════╪══════════╪══════════╪═════╪═══╪══════╪═════╪══════╪═════════╡
    │ 929.89731   ┆ 3        ┆ 1        ┆ 120 ┆ … ┆ 0.04 ┆ 4   ┆ 0    ┆ sqr     │
    │ 931.917313  ┆ 3        ┆ 1        ┆ 120 ┆ … ┆ 0.04 ┆ 4   ┆ 0    ┆ sqr     │
    │ 931.947313  ┆ 3        ┆ 1        ┆ 120 ┆ … ┆ 0.04 ┆ 4   ┆ 0    ┆ sqr     │
    │ 931.957313  ┆ 3        ┆ 1        ┆ 120 ┆ … ┆ 0.04 ┆ 4   ┆ 0    ┆ sqr     │
    │ 931.977313  ┆ 3        ┆ 1        ┆ 120 ┆ … ┆ 0.04 ┆ 4   ┆ 0    ┆ sqr     │
    │ …           ┆ …        ┆ …        ┆ …   ┆ … ┆ …    ┆ …   ┆ …    ┆ …       │
    │ 2733.077838 ┆ 3        ┆ 1        ┆ 180 ┆ … ┆ 0.16 ┆ 1   ┆ 0    ┆ sqr     │
    │ 2733.087838 ┆ 3        ┆ 1        ┆ 180 ┆ … ┆ 0.16 ┆ 1   ┆ 0    ┆ sqr     │
    │ 2733.107838 ┆ 3        ┆ 1        ┆ 180 ┆ … ┆ 0.16 ┆ 1   ┆ 0    ┆ sqr     │
    │ 2733.117838 ┆ 3        ┆ 1        ┆ 180 ┆ … ┆ 0.16 ┆ 1   ┆ 0    ┆ sqr     │
    │ 2733.137838 ┆ 3        ┆ 1        ┆ 180 ┆ … ┆ 0.16 ┆ 1   ┆ 0    ┆ sqr     │
    └─────────────┴──────────┴──────────┴─────┴───┴──────┴─────┴──────┴─────────┘



statemachine dataframe
^^^^^^^^^^^^^^^^^^^^^^^^^
Get statemachine dataframe

.. code-block:: python

    print(stimlog.get_state_machine_dataframe())

- **output of bitbucket version**

.. code-block:: text

    ┌──────────┬───────┬──────────┬──────────┬──────────────┬───────────┐
    │ elapsed  ┆ cycle ┆ newState ┆ oldState ┆ stateElapsed ┆ trialType │
    │ ---      ┆ ---   ┆ ---      ┆ ---      ┆ ---          ┆ ---       │
    │ f64      ┆ f64   ┆ f64      ┆ f64      ┆ f64          ┆ f64       │
    ╞══════════╪═══════╪══════════╪══════════╪══════════════╪═══════════╡
    │ 902.633  ┆ 0.0   ┆ 1.0      ┆ 0.0      ┆ 902.633      ┆ 0.0       │
    │ 904.645  ┆ 0.0   ┆ 2.0      ┆ 1.0      ┆ 2.012        ┆ 0.0       │
    │ 907.656  ┆ 0.0   ┆ 0.0      ┆ 2.0      ┆ 3.01         ┆ 0.0       │
    │ 907.656  ┆ 0.0   ┆ 1.0      ┆ 0.0      ┆ 0.0          ┆ 0.0       │
    │ 909.668  ┆ 0.0   ┆ 2.0      ┆ 1.0      ┆ 2.012        ┆ 0.0       │
    │ …        ┆ …     ┆ …        ┆ …        ┆ …            ┆ …         │
    │ 1801.849 ┆ 0.0   ┆ 0.0      ┆ 2.0      ┆ 3.01         ┆ 0.0       │
    │ 1801.849 ┆ 0.0   ┆ 1.0      ┆ 0.0      ┆ 0.0          ┆ 0.0       │
    │ 1803.862 ┆ 0.0   ┆ 2.0      ┆ 1.0      ┆ 2.012        ┆ 0.0       │
    │ 1806.873 ┆ 0.0   ┆ 3.0      ┆ 2.0      ┆ 3.01         ┆ 0.0       │
    │ 1806.873 ┆ 0.0   ┆ 0.0      ┆ 3.0      ┆ 0.0          ┆ 0.0       │
    └──────────┴───────┴──────────┴──────────┴──────────────┴───────────┘

- **output of github version**

.. code-block:: text


    ┌─────────────┬───────────────────────────┬───────────────────────────┐
    │ time        ┆ state                     ┆ prev_state                │
    │ ---         ┆ ---                       ┆ ---                       │
    │ f64         ┆ str                       ┆ str                       │
    ╞═════════════╪═══════════════════════════╪═══════════════════════════╡
    │ 929.89731   ┆ ('States.SHOW_BLANK', 1)  ┆ ('States.STIM_SELECT', 0) │
    │ 931.907313  ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
    │ 934.917317  ┆ ('States.STIM_SELECT', 0) ┆ ('States.SHOW_STIM', 2)   │
    │ 934.917317  ┆ ('States.SHOW_BLANK', 1)  ┆ ('States.STIM_SELECT', 0) │
    │ 936.91732   ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
    │ …           ┆ …                         ┆ …                         │
    │ 2725.127827 ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
    │ 2728.127831 ┆ ('States.STIM_SELECT', 0) ┆ ('States.SHOW_STIM', 2)   │
    │ 2728.127831 ┆ ('States.SHOW_BLANK', 1)  ┆ ('States.STIM_SELECT', 0) │
    │ 2730.137834 ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
    │ 2733.137838 ┆ ('States.STIM_SELECT', 0) ┆ ('States.SHOW_STIM', 2)   │
    └─────────────┴───────────────────────────┴───────────────────────────┘


stimulus_segment
^^^^^^^^^^^^^^^^^^^^
Get stimulus time segment array: ``Array[float, [S, 2]]``.

.. code-block:: python

    print(stimlog.stimulus_segment)


- **output**

.. code-block:: text

    [[ 904.653     907.678877]
     [ 909.701     912.710125]
     ...
     [2688.334    2691.342914]
     [2693.35     2696.359   ]
     [2698.381    2701.390117]
     [2703.396    2706.404978]
     [2708.428    2711.436947]]


stimulus on-off pulse
^^^^^^^^^^^^^^^^^^^^^^
Get visual stimulation event as square pulse


.. code-block:: python

    vstim = stimlog.stim_square_pulse_event()
    plt.plot(vstim.time, vstim.value)
    plt.show()


profile_dataframe
^^^^^^^^^^^^^^^^^^
Get stim index and trial index dataframe

.. code-block:: python

    print(stimlog.profile_dataframe)

- **output**

.. code-block:: text

    ┌─────────┬──────────┐
    │ i_stims ┆ i_trials │
    │ ---     ┆ ---      │
    │ i64     ┆ i64      │
    ╞═════════╪══════════╡
    │ 35      ┆ 0        │
    │ 22      ┆ 0        │
    │ 11      ┆ 0        │
    │ 25      ┆ 0        │
    │ 61      ┆ 0        │
    │ …       ┆ …        │
    │ 63      ┆ 4        │
    │ 22      ┆ 4        │
    │ 43      ┆ 4        │
    │ 8       ┆ 4        │
    │ 24      ┆ 4        │
    └─────────┴──────────┘


Version-specific attributes
------------------------------
Some attributes/method call are stimpy version specific

Github version only
^^^^^^^^^^^^^^^^^^^^^^^^
- **Photo Indicator dataframe**

.. code-block:: python

    print(stimlog.get_photo_indicator_dataframe())

- **output**

.. code-block:: text

    ┌─────────────┬───────┬──────┬───────────────┬───────┬──────┬────────┬────────┐
    │ time        ┆ state ┆ size ┆ pos           ┆ units ┆ mode ┆ frames ┆ enable │
    │ ---         ┆ ---   ┆ ---  ┆ ---           ┆ ---   ┆ ---  ┆ ---    ┆ ---    │
    │ f64         ┆ bool  ┆ i64  ┆ array[i64, 2] ┆ str   ┆ i64  ┆ i64    ┆ bool   │
    ╞═════════════╪═══════╪══════╪═══════════════╪═══════╪══════╪════════╪════════╡
    │ 929.89731   ┆ false ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 931.917313  ┆ true  ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 934.917317  ┆ false ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 936.92732   ┆ true  ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 939.928324  ┆ false ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ …           ┆ …     ┆ …    ┆ …             ┆ …     ┆ …    ┆ …      ┆ …      │
    │ 2720.12782  ┆ true  ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 2723.117824 ┆ false ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 2725.147827 ┆ true  ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 2728.127831 ┆ false ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    │ 2730.157834 ┆ true  ┆ 35   ┆ [740, 370]    ┆ pix   ┆ 0    ┆ 20     ┆ true   │
    └─────────────┴───────┴──────┴───────────────┴───────┴──────┴────────┴────────┘


- **Log dict dataframe**

.. code-block:: python

    print(stimlog.get_log_dict_dataframe())

- **output**

.. code-block:: text

    ┌─────────────┬──────────┬──────────┬──────────────┬────────────┐
    │ time        ┆ block_nr ┆ trial_nr ┆ condition_nr ┆ trial_type │
    │ ---         ┆ ---      ┆ ---      ┆ ---          ┆ ---        │
    │ f64         ┆ i64      ┆ i64      ┆ i64          ┆ i64        │
    ╞═════════════╪══════════╪══════════╪══════════════╪════════════╡
    │ 929.89731   ┆ 0        ┆ 0        ┆ 35           ┆ 1          │
    │ 934.917317  ┆ 0        ┆ 1        ┆ 22           ┆ 1          │
    │ 939.928324  ┆ 0        ┆ 2        ┆ 11           ┆ 1          │
    │ 944.938331  ┆ 0        ┆ 3        ┆ 25           ┆ 1          │
    │ 949.938338  ┆ 0        ┆ 4        ┆ 61           ┆ 1          │
    │ …           ┆ …        ┆ …        ┆ …            ┆ …          │
    │ 2708.097803 ┆ 4        ┆ 67       ┆ 63           ┆ 1          │
    │ 2713.10781  ┆ 4        ┆ 68       ┆ 22           ┆ 1          │
    │ 2718.117817 ┆ 4        ┆ 69       ┆ 43           ┆ 1          │
    │ 2723.117824 ┆ 4        ┆ 70       ┆ 8            ┆ 1          │
    │ 2728.127831 ┆ 4        ┆ 71       ┆ 24           ┆ 1          │
    └─────────────┴──────────┴──────────┴──────────────┴────────────┘



Stimulus pattern
------------------------------

Refer to :doc:`../api/stimpyp.parser.stimulus`

Current only support :class:`~stimpyp.parser.stimulus.GratingPattern` and :class:`~stimpyp.parser.stimulus.FunctionPattern`

- **example of grating stimulus generator**

.. code-block:: python

    for stim in stimlog.get_stim_pattern().foreach_stimulus(name=True):
         print(stim)

- **output**

.. code-block:: text

    GratingStim(index=0, time=array([904.52436441, 907.52436841]), sf=1, tf=1, direction=0.16)
    GratingStim(index=1, time=array([909.53437141, 912.53537541]), sf=1, tf=1, direction=0.08)
    GratingStim(index=2, time=array([914.55537841, 917.54538241]), sf=1, tf=1, direction=0.04)
    GratingStim(index=3, time=array([919.55538541, 922.54538941]), sf=1, tf=1, direction=0.16)
    GratingStim(index=5, time=array([929.58539941, 932.57540341]), sf=1, tf=1, direction=0.08)
    ...



