import warnings # to debug

import numpy as np
import matplotlib.pyplot as plt

from oneibl.one import ONE
import alf.io

import ibllib.io.extractors
from ibllib.io import spikeglx
import ibllib.plots as iblplots

# To log errors : 
import logging
_logger = logging.getLogger('ibllib')
def _single_test(assertion, str_ok, str_ko):
    if assertion:
        _logger.info(str_ok)
        return True
    else:
        _logger.error(str_ko)
        return False

one = ONE()
eid = one.search(subject='KS005', date_range='2019-08-30', number=1)[0]
eid = one.search(subject='KS016', date_range='2019-12-05', number=1)[0]
# eid = one.search(subject='CSHL_020', date_range='2019-12-04', number=5)[0]

one.alyx.rest('sessions', 'read', id=eid)['task_protocol']

one.list(eid)
dtypes = [
         '_spikeglx_sync.channels',
         '_spikeglx_sync.polarities',
         '_spikeglx_sync.times',
         '_iblrig_taskSettings.raw',
         '_iblrig_taskData.raw',
         '_iblrig_encoderEvents.raw',
         '_iblrig_encoderPositions.raw',
         '_iblrig_encoderTrialInfo.raw',
]

files = one.load(eid, dataset_types=dtypes, download_only=True)
sess_path = alf.io.get_session_path(files[0])

chmap = ibllib.io.extractors.ephys_fpga.CHMAPS['3B']['nidq']
# chmap = ibllib.io.extractors.ephys_fpga.CHMAPS['3A']['ap']

"""get the sync pulses dealing with 3A and 3B revisions"""
if next(sess_path.joinpath('raw_ephys_data').glob('_spikeglx_sync.*'), None):
    # if there is nidq sync it's a 3B session
    sync_path = sess_path.joinpath(r'raw_ephys_data')
else:  # otherwise it's a 3A
    # TODO find the main sync probe
    # sync_path = sess_path.joinpath(r'raw_ephys_data', 'probe00')
    pass
sync = alf.io.load_object(sync_path, '_spikeglx_sync', short_keys=True)

"""get the wheel data for both fpga and bpod"""
fpga_wheel = ibllib.io.extractors.ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
bpod_wheel = ibllib.io.extractors.training_wheel.get_wheel_data(sess_path, save=False)

"""get the behaviour data for both fpga and bpod"""


# -- Out FPGA : 
# dict_keys(['ready_tone_in', 'error_tone_in', 'valve_open', 'stim_freeze', 'stimOn_times',
# 'iti_in', 'goCue_times', 'feedback_times', 'intervals', 'response_times'])
ibllib.io.extractors.ephys_trials.extract_all(sess_path, save=True)
fpga_behaviour = ibllib.io.extractors.ephys_fpga.extract_behaviour_sync(
    sync, output_path=sess_path.joinpath('alf'), chmap=chmap, save=True, display=True)

# TODO valve open partout
# -- Out BPOD :
# dict_keys(['feedbackType', 'contrastLeft', 'contrastRight', 'probabilityLeft',
# 'session_path', 'choice', 'rewardVolume', 'feedback_times', 'stimOn_times', 'intervals',
# 'response_times', 'camera_timestamps', 'goCue_times', 'goCueTrigger_times',
# 'stimOnTrigger_times', 'included'])
bpod_behaviour = ibllib.io.extractors.biased_trials.extract_all(sess_path, save=False)

"""get the sync between behaviour and bpod"""
bpod_offset = ibllib.io.extractors.ephys_fpga.align_with_bpod(sess_path)


## -----   PLOTS    -----
# plt.figure(1)
# fix, axes = plt.subplots(nrows=2, sharex='all', sharey='all')
# axes[0].plot(t, pos), axes[0].title.set_text('Extracted')
# axes[0].plot(fpga_wheel['re_ts'], fpga_wheel['re_pos']), axes[0].title.set_text('FPGA')
# axes[0].plot(bpod_wheel['re_ts'] + bpod_offset, bpod_wheel['re_pos'])
# axes[1].plot(bpod_wheel['re_ts'] + bpod_offset, bpod_wheel['re_pos'])
# axes[1].title.set_text('Bpod')

# plt.figure(2)
# plt.plot(fpga_behaviour['intervals'][:, 0], bpod_behaviour['stimOn_times'] -
#         fpga_behaviour['stimOn_times'] + bpod_offset)

# plt.figure(3)
# plt.plot(fpga_behaviour['stimOn_times'] - fpga_behaviour['intervals'][:, 0] )


# ------------------------------------------------------
#          Start the QC part (Ephys only)
# ------------------------------------------------------

# Make a bunch gathering all trial QC
from brainbox.core import Bunch

size_stimOn_goCue = [np.size(fpga_behaviour['stimOn_times']), np.size(fpga_behaviour['goCue_times'])]
size_response_goCue = [np.size(fpga_behaviour['response_times']), np.size(fpga_behaviour['goCue_times'])]


trials_qc = Bunch({
    # TEST  StimOn and GoCue should all be within a very small tolerance of each other
    #       1. check for non-Nans
    'stimOn_times_nan': ~np.isnan(fpga_behaviour['stimOn_times']),  
    'goCue_times_nan': ~np.isnan(fpga_behaviour['goCue_times']),
    #       2. check if closeby value
    'stimOn_times_goCue_times_diff': np.all(fpga_behaviour['goCue_times'] - fpga_behaviour['stimOn_times']) < 0.010,
    # TEST  Response times (from session start) should be increasing continuously
    #       Note: RT are not durations but time stamps from session start
    #       1. check for non-Nans
    'response_times_nan': ~np.isnan(fpga_behaviour['response_times']),
    #       2. check for positive increase
    'response_times_increase': np.diff(np.append([0], fpga_behaviour['response_times'])) > 0,
    # TEST  Response times (from goCue) should be positive
    'response_times_goCue_times_diff': fpga_behaviour['response_times'] - fpga_behaviour['goCue_times'] > 0,

})

session_qc_test = Bunch({
    # TEST  StimOn and GoCue should be of similar size
    'stimOn_times_goCue_times_size': np.size(np.unique(size_stimOn_goCue)) == 1,
    # TEST  Response times and goCue  should be of similar size
    'response_times_goCue_times_size': np.size(np.unique(size_response_goCue)) == 1,
})

# Test output at session level ## OLIVIER TODO THIS BUGS
import pandas as pd
pd_trials_qc = pd.DataFrame.from_dict(trials_qc)
session_qc = {k:np.all(trials_qc[k]) for k in trials_qc}


## ========================TODO DELETE BELOW =================================
# TEST  StimOn and GoCue should all be within a very small tolerance of each other
#       1. check for non-Nans
from brainbox.core import Bunch
trials_qc = Bunch({
    'stimOn_times_nan': ~np.isnan(fpga_behaviour['stimOn_times']),
    'goCue_times_nan': ~np.isnan(fpga_behaviour['goCue_times']),
})

import pandas as pd

pd_trials_qc = pd.DataFrame.from_dict(trials_qc)

session_qc = {k:np.all(trials_qc[k]) for k in trials_qc}


fpga_behaviour['valve_open'].size
fpga_behaviour['ready_tone_in'].size


session_qc['stimOn_times_nan']

_single_test(not np.any(np.isnan(fpga_behaviour['stimOn_times'])),
             '(Ephys) Test Pass   : stimOn_times without Nans',
             '(Ephys) !! ERROR !! : stimOn_times contains Nans')

_single_test(not np.any(np.isnan(fpga_behaviour['goCue_times'])),
             '(Ephys) Test Pass   : goCue_times without Nans',
             '(Ephys) !! ERROR !! : goCue_times contains Nans')

#       2. check for similar size
array_size = np.zeros((2, 1))
array_size[0] = np.size(fpga_behaviour['stimOn_times'])
array_size[1] = np.size(fpga_behaviour['goCue_times'])

_single_test(np.size(np.unique(array_size)) == 1,
             '(Ephys) Test Pass   : size stimOn_times == goCue_times',
             '(Ephys) !! ERROR !! : size stimOn_times != goCue_times')

#       3. check if closeby value
dtimes_stimOn_goCue = {}
dtimes_stimOn_goCue = fpga_behaviour['goCue_times'] - fpga_behaviour['stimOn_times'] # the goCue tone should be after the stim on

_single_test(np.all(dtimes_stimOn_goCue < 0.010), # min value should be ~600us, could be replaced by 0.001
             '(Ephys) Test Pass   : stimOn_times & goCue_times closeby',
             '(Ephys) !! ERROR !! : stimOn_times & goCue_times too far')

# TEST  Response times (from session start) should be increasing continuously
#       Note: RT are not durations but time stamps from session start
#       1. check for non-Nans
_single_test(not np.any(np.isnan(fpga_behaviour['response_times'])),
             '(Ephys) Test Pass   : response_times without Nans',
             '(Ephys) !! ERROR !! : response_times contains Nans')

#       2. check for positive increase
_single_test(np.all(np.diff(fpga_behaviour['response_times']) > 0),
             '(Ephys) Test Pass   : RT diff positive',
             '(Ephys) !! ERROR !! : RT diff negative')

# TEST  Response times (from goCue) should be positive
#       1. check for similar size
array_size = np.zeros((2, 1))
array_size[0] = np.size(fpga_behaviour['response_times'])
array_size[1] = np.size(fpga_behaviour['goCue_times'])

_single_test(np.size(np.unique(array_size)) == 1,
             '(Ephys) Test Pass   : size response_times == goCue_times',
             '(Ephys) !! ERROR !! : size response_times != goCue_times')

#       2. check if positive
_single_test(np.all(fpga_behaviour['response_times'] - fpga_behaviour['goCue_times'] > 0),
             '(Ephys) Test Pass   : RT from goCue positive',
             '(Ephys) !! ERROR !! : RT from goCue negative')             
## ========================TODO DELETE ABOVE =================================

# TEST  Start of iti_in should be within a very small tolerance of the stim off
# TODO QUESTION OLIVIER: How do I get stim off times ?
# fpga_behaviour['stim_freeze']

# TEST  Wheel should not move xx amount of time (quiescent period) before go cue
#       Wheel should move before feedback
# TODO ingest code from Michael S : https://github.com/int-brain-lab/ibllib/blob/brainbox/brainbox/examples/count_wheel_time_impossibilities.py 

# TEST  No frame2ttl change between stim off and go cue
# TODO QUESTION OLIVIER: How do I get stim off times ?

# TEST  Delay between valve and stim off should be 1s
# TODO QUESTION OLIVIER: How do I get stim off times ?
# fpga_behaviour['valve_open']

# ------------------------------------------------------
#          Start the QC part (Bpod+Ephys)
# ------------------------------------------------------

# TEST  Compare times from the bpod behaviour extraction to the Ephys extraction
dbpod_fpga = {}
for k in ['goCue_times', 'stimOn_times']:
    dbpod_fpga[k] = bpod_behaviour[k] - fpga_behaviour[k] + bpod_offset
    # we should use the diff from trial start for a more accurate test but this is good enough for now
    assert np.all(dbpod_fpga[k] < 0.05)

# ------------------------------------------------------
#          Start the QC PART (Bpod only)
# ------------------------------------------------------

# TEST  StimOn, StimOnTrigger, GoCue and GoCueTrigger should all be within a very small tolerance of each other
#       1. check for non-Nans
assert not np.any(np.isnan(bpod_behaviour['stimOn_times']))
assert not np.any(np.isnan(bpod_behaviour['goCue_times']))
assert not np.any(np.isnan(bpod_behaviour['stimOnTrigger_times']))
assert not np.any(np.isnan(bpod_behaviour['goCueTrigger_times']))

#       2. check for similar size
array_size = np.zeros((4, 1))
array_size[0] = np.size(bpod_behaviour['stimOn_times'])
array_size[1] = np.size(bpod_behaviour['goCue_times'])
array_size[2] = np.size(bpod_behaviour['stimOnTrigger_times'])
array_size[3] = np.size(bpod_behaviour['goCueTrigger_times'])
assert np.size(np.unique(array_size)) == 1