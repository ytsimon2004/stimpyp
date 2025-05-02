import unittest

from stimpyp.dataset.treadmill import load_example_data
from stimpyp.parser import StimlogBase, SessionInfo


class TestStimlog(unittest.TestCase):
    stimlog: StimlogBase

    @classmethod
    def setUpClass(cls):
        cls.stimlog = load_example_data('stimpy-bit').get_stimlog()

    def test_session_trials(self):
        exp = {
            'light_bas': SessionInfo(name='light_bas', time=(0.007, 900)),
            'dark': SessionInfo(name='dark', time=(900, 1810)),
            'light_end': SessionInfo(name='light_end', time=(1810, 2711.532)),
            'all': SessionInfo(name='all', time=(0.007, 2711.532))
        }

        res = self.stimlog.session_trials()

        self.assertDictEqual(res, exp)


if __name__ == '__main__':
    unittest.main()
