import pytest

from stimpyp import RiglogData, PyVlog
from ._dataset import load_example_riglog


class TestRiglogBit:
    riglog: RiglogData

    @classmethod
    def setup_class(cls):
        cls.riglog = load_example_riglog('stimpy-bit', stim_type='sftfdir')

    def test_config(self):
        assert self.riglog.log_config == {
            'codes': {'act0': 21,
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
            'commit_hash': '6d30281',
            'fields': ('code', 'time received', 'duino time', 'value'),
            'source_version': 'stimpy-bit',
            'version': 0.3
        }

    def test_event(self):
        img_event = self.riglog.imaging_event
        assert img_event.start_time == pytest.approx(0.566)
        assert img_event.end_time == pytest.approx(3607.018)

    def test_camera_event(self):
        wfield_event = self.riglog.camera_event['1P_cam']
        assert wfield_event.n_pulses == 81526


class TestRiglogGit:
    riglog: RiglogData

    @classmethod
    def setup_class(cls):
        cls.riglog = load_example_riglog('stimpy-git', stim_type='sftfdir')

    def test_config(self):
        assert self.riglog.log_config == {
            'commit_hash': '88c4705',
            'codes': {'screen': 0,
                      'imaging': 1,
                      'encoder': 2,
                      'licks': 3,
                      'button': 4,
                      'reward': 5,
                      'laps': 6,
                      'cam1': 7,
                      'cam2': 8,
                      'cam3': 9,
                      'act0': 10,
                      'act1': 11,
                      'opto': 12},
            'fields': ('code', 'time received', 'duino time', 'value'),
            'source_version': 'stimpy-git'
        }

    def test_event(self):
        img_event = self.riglog.imaging_event
        assert img_event.start_time == pytest.approx(0.589)
        assert img_event.end_time == pytest.approx(3596.476)

    def test_camera_event(self):
        wfield_event = self.riglog.camera_event['1P_cam']
        assert wfield_event.n_pulses == 81327


class TestRiglogPyV:
    log: PyVlog

    @classmethod
    def setup_class(cls):
        cls.log = load_example_riglog('pyvstim', stim_type='circular')

    def test_config(self):
        assert self.log.log_config == {
            'version': '1.4',
            'commit_hash': 'af97b40',
            'rig_version': '0.3',
            'source_version': 'pyvstim'
        }

    def test_event(self):
        screen_event = self.log.screen_event
        assert screen_event.start_time == pytest.approx(11.194)
        assert screen_event.end_time == pytest.approx(1147.623)

    def test_camera_event(self):
        wfield_event = self.log.camera_event['1P_cam']
        assert wfield_event.n_pulses == 9622
