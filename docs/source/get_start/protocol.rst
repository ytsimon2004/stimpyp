Protocol
==========
Stimpy protocol file

Refer to :class:`~stimpyp.parser.stimpy_core.StimpyProtocol`


**Example of bitbucket version**

.. code-block:: text

    # general parameters
    controller = VisualExpController
    stimulusType = gratings
    nTrials = 5
    shuffle = True
    startBlankDuration = 900
    blankDuration = 2
    endBlankDuration = 900
    texture = sqr
    mask = None

    # stimulus conditions
    n     dur  xc   yc   c    sf      ori    flick  width height evolveParams
    1     3    0    0    1    0.04    0      0      200    200    {'phase':['linear',1]}
    2     3    0    0    1    0.04    30     0      200    200    {'phase':['linear',1]}
    3     3    0    0    1    0.04    60     0      200    200    {'phase':['linear',1]}
    4     3    0    0    1    0.04    90     0      200    200    {'phase':['linear',1]}
    5     3    0    0    1    0.04    120    0      200    200    {'phase':['linear',1]}
    6     3    0    0    1    0.04    150    0      200    200    {'phase':['linear',1]}
    7     3    0    0    1    0.04    180    0      200    200    {'phase':['linear',1]}
    8     3    0    0    1    0.04    210    0      200    200    {'phase':['linear',1]}
    9     3    0    0    1    0.04    240    0      200    200    {'phase':['linear',1]}
    10    3    0    0    1    0.04    270    0      200    200    {'phase':['linear',1]}
    11    3    0    0    1    0.04    300    0      200    200    {'phase':['linear',1]}
    12    3    0    0    1    0.04    330    0      200    200    {'phase':['linear',1]}
    ...


**Example of github version**

.. code-block:: text

    # general parameters
    controller = user.VisualExperimentController
    displayType = psychopy
    background = 0.5

    stimulusType = gratings
    nTrials = 5
    shuffle = True
    blankDuration = 2
    startBlankDuration = 900
    endBlankDuration = 900
    mask = None

    # stimulus conditions
    visual_stimuli =
    # ^^^^^^^^^^^^^^ [1]
        n      dur     xc   yc   c    sf    tf   ori      width  height pattern
         1-12  3       0    0    1    0.04  1    {i}*30   200    200    sqr
        13-24  3       0    0    1    0.08  1    {i}*30   200    200    sqr
        25-36  3       0    0    1    0.16  1    {i}*30   200    200    sqr
        37-48  3       0    0    1    0.04  4    {i}*30   200    200    sqr
        49-60  3       0    0    1    0.08  4    {i}*30   200    200    sqr
        61-72  3       0    0    1    0.04  4    {i}*30   200    200    sqr

1. Use for determine if the protocol file is which **stimpy version**


init protocol
----------------

**Infer from riglog**

.. code-block:: python

    from stimpyp.parser import load_riglog

    file = ...  # riglog file path or riglog directory path
    riglog = load_riglog(file)
    protocol = riglog.get_protocol()


**Load file**

.. code-block:: python

    from stimpyp.parser import load_protocol
    file = ...  # protocol file path
    protocol = load_protocol(file)


protocol options
---------------------

.. code-block:: python

    print(protocol.options)

- **output**

.. code-block:: text

    {
    │   'controller': 'user.VisualExperimentController',
    │   'displayType': 'psychopy',
    │   'background': 0.5,
    │   'stimulusType': 'gratings',
    │   'nTrials': 5,
    │   'shuffle': 'True',
    │   'blankDuration': 2,
    │   'startBlankDuration': 900,
    │   'endBlankDuration': 900,
    │   'mask': 'None',
    │   'visual_stimuli': ''
    }


as dataframe
-------------------

.. code-block:: python

    print(protocol.visual_stimuli_dataframe)

- **output**

.. code-block:: text

    ┌─────┬─────┬─────┬─────┬───┬─────┬───────┬────────┬─────────┐
    │ n   ┆ dur ┆ xc  ┆ yc  ┆ … ┆ ori ┆ width ┆ height ┆ pattern │
    │ --- ┆ --- ┆ --- ┆ --- ┆   ┆ --- ┆ ---   ┆ ---    ┆ ---     │
    │ i64 ┆ i64 ┆ i64 ┆ i64 ┆   ┆ i64 ┆ i64   ┆ i64    ┆ str     │
    ╞═════╪═════╪═════╪═════╪═══╪═════╪═══════╪════════╪═════════╡
    │ 1   ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 0   ┆ 200   ┆ 200    ┆ sqr     │
    │ 2   ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 30  ┆ 200   ┆ 200    ┆ sqr     │
    │ 3   ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 60  ┆ 200   ┆ 200    ┆ sqr     │
    │ 4   ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 90  ┆ 200   ┆ 200    ┆ sqr     │
    │ 5   ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 120 ┆ 200   ┆ 200    ┆ sqr     │
    │ …   ┆ …   ┆ …   ┆ …   ┆ … ┆ …   ┆ …     ┆ …      ┆ …       │
    │ 68  ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 210 ┆ 200   ┆ 200    ┆ sqr     │
    │ 69  ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 240 ┆ 200   ┆ 200    ┆ sqr     │
    │ 70  ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 270 ┆ 200   ┆ 200    ┆ sqr     │
    │ 71  ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 300 ┆ 200   ┆ 200    ┆ sqr     │
    │ 72  ┆ 3   ┆ 0   ┆ 0   ┆ … ┆ 330 ┆ 200   ┆ 200    ┆ sqr     │
    └─────┴─────┴─────┴─────┴───┴─────┴───────┴────────┴─────────┘