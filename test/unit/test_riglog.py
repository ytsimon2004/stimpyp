from stimpyp.dataset.treadmill import load_example_data


def riglog():
    return load_example_data('stimpy-bit')


def test_config():
    assert riglog().log_config == {
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
        'commit_hash': '08701fb',
        'fields': ('code', 'time received', 'duino time', 'value'),
        'source_version': 'stimpy-bit',
        'version': 0.3
    }


def test_event():
    x = riglog().camera_event['1P_cam'].n_pulses == 59870
    print(x)


test_event()
