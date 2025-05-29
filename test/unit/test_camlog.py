from ._dataset import load_example_camlog


class TestLabCamLog:

    @classmethod
    def setup_class(cls):
        cls.camlog = load_example_camlog('labcams')

    def test_comment_info(self):
        assert self.camlog.comment_info == {
            'Camera': 'facecam log file',
            'Date': '15-03-2021',
            'labcams version': '0.2',
            'Log header': 'frame_id,timestamp'
        }

    def test_repr(self):
        print(self.camlog)

    def test_time_info(self):
        assert isinstance(self.camlog.time_info, list)
        assert isinstance(self.camlog.time_info[0], str)

    def test_dataframe(self):
        print(self.camlog.to_polars())


class TestPyCamLog:
    @classmethod
    def setup_class(cls):
        cls.camlog = load_example_camlog('pycams')

    def test_comment_info(self):
        assert self.camlog.comment_info == {
            'Commit hash': '50082af',
            'Log header': 'frame_id,timestamp'
        }

    def test_time_info(self):
        assert self.camlog.time_info is None

    def test_dataframe(self):
        print(self.camlog.to_polars())
