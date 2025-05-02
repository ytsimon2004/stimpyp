import unittest

from stimpyp.dataset.treadmill import load_example_data
from stimpyp.parser import PyVlog


class TestLogParser(unittest.TestCase):
    log: PyVlog

    @classmethod
    def setUpClass(cls):
        cls.log = load_example_data('pyvstim', 'circular')

    def test_source_version(self):
        self.assertEqual(self.log.version, 'pyvstim')

    def test_version(self):
        self.assertEqual(self.log.log_config['rig_version'], '0.3')

    def test_commit_hash(self):
        self.assertEqual(self.log.log_config['commit_hash'], 'af97b40')

    def test_codes(self):
        exp = {0: 'screen',
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
               22: 'act1'}

        self.assertDictEqual(self.log.log_info, exp)

    def test_csv_fields(self):
        res = ['code', 'time received', 'duino time', 'value']
        self.assertListEqual(self.log.log_header[1], res)


if __name__ == '__main__':
    unittest.main()
