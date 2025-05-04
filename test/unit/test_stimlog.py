from stimpyp.dataset.treadmill import load_example_data


def stimlog():
    return load_example_data('stimpy-git', stim_type='sftfdir').get_stimlog()


def test_config():
    stim = stimlog()
    assert stim.config == {'commit_hash': '6d30281', 'missed_frames': 0}
    assert stim.log_info == {10: 'vstim', 20: 'stateMachine'}
    assert stim.log_header == {
        10: ['code', 'presentTime', 'iStim', 'iTrial', 'photo', 'contrast', 'ori', 'sf', 'phase', 'stim_idx'],
        20: ['code', 'elapsed', 'cycle', 'newState', 'oldState', 'stateElapsed', 'trialType']}


df = stimlog().profile_dataframe
print(df)
