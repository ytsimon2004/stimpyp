from rich.pretty import pprint

from stimpyp.dataset.treadmill import load_example_data


def protocol():
    return load_example_data('stimpy-git', stim_type='sftfdir').get_protocol()


print(protocol().name)
pprint(protocol().visual_stimuli_dataframe)
