import polars as pl

from stimpyp.parser import Stimlog, StimlogGit, StimlogPyVStim
from ._dataset import load_example_data


class TestStimlogBit:
    stimlog: Stimlog

    # noinspection PyTypeChecker
    @classmethod
    def setup_class(cls):
        cls.stimlog = load_example_data('stimpy-bit', stim_type='sftfdir').get_stimlog()

    def test_config(self):
        assert self.stimlog.config == {'commit_hash': '6d30281', 'missed_frames': 0}
        assert self.stimlog.log_info == {10: 'vstim', 20: 'stateMachine'}
        assert self.stimlog.log_header == {
            10: ['code', 'presentTime', 'iStim', 'iTrial', 'photo', 'contrast', 'ori', 'sf', 'phase', 'stim_idx'],
            20: ['code', 'elapsed', 'cycle', 'newState', 'oldState', 'stateElapsed', 'trialType']
        }

    def test_dataframe(self):
        df_visual = self.stimlog.get_visual_stim_dataframe()
        df_statemachine = self.stimlog.get_state_machine_dataframe()
        df_profile = self.stimlog.profile_dataframe

        assert isinstance(df_visual, pl.DataFrame)
        assert df_visual.columns == ['presentTime', 'iStim', 'iTrial', 'photo', 'contrast', 'ori', 'sf', 'phase', 'stim_idx']

        assert isinstance(df_statemachine, pl.DataFrame)
        assert df_statemachine.columns == ['elapsed', 'cycle', 'newState', 'oldState', 'stateElapsed', 'trialType']

        assert isinstance(df_profile, pl.DataFrame)
        assert df_profile.columns == ['i_stims', 'i_trials']

    def test_stimulus_generator(self):
        for stim in self.stimlog.get_stim_pattern().foreach_stimulus(name=True):
            assert hasattr(stim, 'index')
            assert hasattr(stim, 'time')
            assert hasattr(stim, 'sf')
            assert hasattr(stim, 'tf')
            assert hasattr(stim, 'direction')
            assert len(stim.time) == 2
            assert stim.time[0] < stim.time[1]


class TestStimlogGit:
    stimlog: StimlogGit

    # noinspection PyTypeChecker
    @classmethod
    def setup_class(cls):
        cls.stimlog = load_example_data('stimpy-git', stim_type='sftfdir').get_stimlog()

    def test_config(self):
        assert self.stimlog.config == {
            'commit_hash': '88c4705',
            'end_time': 3635.080104,
            'format': ['source_id', 'time', 'source_infos'],
            'log_name': 'stimpy_main_logger',
            'missed_frames': 247,
            'rig_trigger': ('imaging', 0.0),
            'start_time': 27.971045,
            'tag': "['']"
        }
        assert self.stimlog.log_info == {0: 'Gratings', 1: 'PhotoIndicator', 2: 'StateMachine', 3: 'LogDict'}
        assert self.stimlog.log_header == {
            0: ['duration', 'contrast', 'ori', 'phase', 'pos', 'size', 'flick', 'interpolate', 'mask',
                'sf', 'tf', 'opto', 'pattern'],
            1: ['state', 'size', 'pos', 'units', 'mode', 'frames', 'enabled'],
            2: ['state', 'prev_state'],
            3: ['block_nr', 'trial_nr', 'condition_nr', 'trial_type']
        }

    def test_dataframe(self):
        df_visual = self.stimlog.get_visual_stim_dataframe()
        df_statemachine = self.stimlog.get_state_machine_dataframe()
        df_photo = self.stimlog.get_photo_indicator_dataframe()
        df_logdict = self.stimlog.get_log_dict_dataframe()
        df_profile = self.stimlog.profile_dataframe

        assert isinstance(df_visual, pl.DataFrame)
        assert df_visual.columns == ['time', 'duration', 'contrast', 'ori', 'phase', 'pos', 'size',
                                     'flick', 'interpolate', 'mask', 'sf', 'tf', 'opto', 'pattern']

        assert isinstance(df_statemachine, pl.DataFrame)
        assert df_statemachine.columns == ['time', 'state', 'prev_state']

        assert isinstance(df_photo, pl.DataFrame)
        assert df_photo.columns == ['time', 'state', 'size', 'pos', 'units', 'mode', 'frames', 'enable']

        assert isinstance(df_logdict, pl.DataFrame)
        assert df_logdict.columns == ['time', 'block_nr', 'trial_nr', 'condition_nr', 'trial_type']

        assert isinstance(df_profile, pl.DataFrame)
        assert df_profile.columns == ['i_stims', 'i_trials']

    def test_stimulus_generator(self):
        for stim in self.stimlog.get_stim_pattern().foreach_stimulus(name=True):
            assert hasattr(stim, 'index')
            assert hasattr(stim, 'time')
            assert hasattr(stim, 'sf')
            assert hasattr(stim, 'tf')
            assert hasattr(stim, 'direction')
            assert len(stim.time) == 2
            assert stim.time[0] < stim.time[1]

    def test_lazy_load(self):
        # lazy_load_stimlog()
        ...


class TestStimlogPyV:
    stimlog: StimlogPyVStim

    @classmethod
    def setup_class(cls):
        cls.stimlog = load_example_data('pyvstim', stim_type='circular').get_stimlog()

    def test_config(self):
        assert self.stimlog.config == {
            'commit_hash': 'af97b40',
            'rig_version': '0.3',
            'source_version': 'pyvstim',
            'version': '1.4'
        }

        assert self.stimlog.log_info == {
            0: 'screen',
            1: 'imaging',
            2: 'position',
            3: 'lick',
            4: 'reward',
            5: 'lap',
            6: 'cam1',
            7: 'cam2',
            8: 'cam3',
            10: 'vstim',
            21: 'act0',
            22: 'act1'
        }

        assert self.stimlog.log_header == {
            0: ['code', 'time received', 'duino time', 'value'],
            1: ['code', 'time received', 'duino time', 'value'],
            2: ['code', 'time received', 'duino time', 'value'],
            3: ['code', 'time received', 'duino time', 'value'],
            4: ['code', 'time received', 'duino time', 'value'],
            5: ['code', 'time received', 'duino time', 'value'],
            6: ['code', 'time received', 'duino time', 'value'],
            7: ['code', 'time received', 'duino time', 'value'],
            8: ['code', 'time received', 'duino time', 'value'],
            10: ['code', 'presentTime', 'iStim', 'iTrial', 'iFrame', 'blank', 'contrast', 'posx', 'posy', 'apx', 'apy',
                 'indicatorFlag'],
            21: ['code', 'time received', 'duino time', 'value'],
            22: ['code', 'time received', 'duino time', 'value']}

    def test_dataframe(self):
        df_visual = self.stimlog.get_visual_stim_dataframe()
        df_profile = self.stimlog.profile_dataframe

        assert isinstance(df_visual, pl.DataFrame)
        assert df_visual.columns == [
            'presentTime',
            'iStim',
            'iTrial',
            'iFrame',
            'blank',
            'contrast',
            'posx',
            'posy',
            'apx',
            'apy',
            'indicatorFlag'
        ]

        assert isinstance(df_profile, pl.DataFrame)
        assert df_profile.columns == ['i_stims', 'i_trials']
