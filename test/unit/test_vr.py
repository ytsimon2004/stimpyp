import unittest

from stimpyp import RiglogData, PyGameLinearStimlog, WorldMapInfo


@unittest.skip("Not yet implemented")
class TestVRTask(unittest.TestCase):
    file = ...

    riglog: RiglogData
    stimlog: PyGameLinearStimlog

    @classmethod
    def setUpClass(cls):
        cls.riglog = RiglogData(root_path=cls.file)
        cls.stimlog = cls.riglog.get_pygame_stimlog()

    def test_session_trials(self):
        print(self.stimlog.session_trials())

    def test_virtual_length(self):
        print(self.stimlog.get_virtual_length())

    def test_virtual_lap_event(self):
        print(self.stimlog.virtual_lap_event)

    def test_virtual_landmarks(self):
        print(self.stimlog.get_landmarks())


@unittest.skip("Not yet implemented")
class TestWorldMap(unittest.TestCase):
    file = ...

    world: WorldMapInfo

    @classmethod
    def setUpClass(cls):
        riglog = RiglogData(root_path=cls.file)
        cls.world = riglog.get_worldmap()

    def test_world_map(self):
        print(self.world.world_map)

    def test_info_pos(self):
        print(self.world.info_pos)

    def test_raw_map_grid(self):
        """Get the raw map grid as characters"""
        world_map = self.world.world_map

        # Convert numeric codes back to characters
        for row in world_map:
            row_str = ''.join(chr(code) if code != 0 else ' ' for code in row)
            print(row_str)
