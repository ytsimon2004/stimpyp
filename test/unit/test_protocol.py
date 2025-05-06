import polars as pl

from stimpyp.parser import StimpyProtocol, PyVProtocol
from ._dataset import load_example_data


class TestProtocolBit:
    protocol: StimpyProtocol

    @classmethod
    def setup_class(cls):
        cls.protocol = load_example_data('stimpy-bit', stim_type='sftfdir').get_protocol()

    def test_options(self):
        assert self.protocol.version == 'stimpy-bit'
        assert self.protocol.options == {
            'controller': 'VisualExpController',
            'stimulusType': 'gratings',
            'nTrials': 5,
            'shuffle': 'True',
            'startBlankDuration': 900,
            'blankDuration': 2,
            'endBlankDuration': 900,
            'texture': 'sqr',
            'mask': 'None'
        }

    def test_dataframe(self):
        df = self.protocol.visual_stimuli_dataframe
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ['n', 'dur', 'xc', 'yc', 'c', 'sf', 'ori', 'flick', 'width', 'height', 'evolveParams']


class TestProtocolGit:
    protocol: StimpyProtocol

    @classmethod
    def setup_class(cls):
        cls.protocol = load_example_data('stimpy-git', stim_type='sftfdir').get_protocol()

    def test_options(self):
        assert self.protocol.options == {
            'controller': 'user.VisualExperimentController',
            'displayType': 'psychopy',
            'background': 0.5,
            'stimulusType': 'gratings',
            'nTrials': 5,
            'shuffle': 'True',
            'blankDuration': 2,
            'startBlankDuration': 900,
            'endBlankDuration': 900,
            'mask': 'None',
            'visual_stimuli': ''  # due to space line
        }

    def test_dataframe(self):
        df = self.protocol.visual_stimuli_dataframe
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ['n', 'dur', 'xc', 'yc', 'c', 'sf', 'tf', 'ori', 'width', 'height', 'pattern']


class TestProtocolPyV:
    protocol: PyVProtocol

    @classmethod
    def setup_class(cls):
        cls.protocol = load_example_data('pyvstim', stim_type='circular').get_protocol()

    def test_options(self):
        assert self.protocol.options == {
            'StimulusType': 'retinoCircling',
            'PicsFolder': 'contrast_reversal',
            'PicsNameFormat': 'stim{0:03d}_frame{1:03d}.png',
            'BlankDuration': 6,
            'DecimationRatio': 10,
            'WindowType': 'rect',
            'ApertureMask': 'circle',
            'LoopBack': 'true',
            'nTrials': 30,
            'Interpolate': 'False',
            'Shuffle': False
        }

    def test_dataframe(self):
        df = self.protocol.visual_stimuli_dataframe
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ['n', 'dur', 'len', 'pad', 'c', 'width', 'xc', 'yc', 'apwidth', 'apheight', 'apxc', 'apyc']

    def test_loop_expression(self):
        expr = self.protocol.get_loops_expr()
        assert expr.expr == ["circle(8*exp.refreshRate,50,output='x')", "circle(8*exp.refreshRate,34,output='y')"]
        assert expr.n_cycles == [4]
        assert expr.n_blocks == 1
