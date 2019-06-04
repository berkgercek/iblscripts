import numpy as np
# TRAINING PARAMETERS ESTIMATIONS
N_LABS = np.array([10, 12])  # 10-12 experimental labs
N_MICE_PER_BATCH = np.array([7, 9])  # 7-9 mice at a time
N_TRAINING_DAYS_PER_YEAR = round(365*5/7)  # 261 days per year training
# 2 Go/camera/hour of recording at 25 Hz, cropped, mpeg compression, 850/570 framesize, 2 cams
# in practice we have 424Mb for 48 mins compressed at 29
# in practice we have 1320 for 70 mins compressed at 23
SIZE_ALF_DATA_PER_TRAINING_SESSION_GB = 20 / 1024
SIZE_TRAINING_VIDEO_HOURLY_GB = 2  # size for all cameras (one)
TRAINING_SESSION_DURATION_HOURS = 1
# EPHYS PARAMETERS ESTIMATIONS
RECORDED_MICE_PER_BATCH = np.array([2, 3])  # mice recorded per batch
TRAINING_CYCLE_DURATION_DAYS = 7 * 8  # 8 weeks from training to recording
RECORDINGS_PER_MOUSE = np.array([4, 6])  # recordings sessions per mouse
EPHYS_SESSION_DURATION_HOURS = 1
SIZE_RECORDING_VIDEO_HOURLY_GB = np.array([4, 6]) * 3
SIZE_ALF_DATA_PER_RECORDING_SESSION_GB = 70 / 1024 * 3  # NB: assumes 3 versions of spike sorting

Tb = {
    'training_alf': None,
    'training videos': None,
    'ephys_raw': None,
    'ephys_videos': None,
    'ephys_alf': None,
    'histology': None,
}
# Training sessions
n_ses_per_year = N_TRAINING_DAYS_PER_YEAR * N_LABS * N_MICE_PER_BATCH
Tb['training_videos'] = n_ses_per_year * SIZE_TRAINING_VIDEO_HOURLY_GB * TRAINING_SESSION_DURATION_HOURS / 1024
Tb['training_alf'] = SIZE_ALF_DATA_PER_TRAINING_SESSION_GB * n_ses_per_year / 1024

# Ephys
n_mice_rec_per_year = RECORDED_MICE_PER_BATCH * np.round(365 / TRAINING_CYCLE_DURATION_DAYS * N_LABS)  # IBL mice recorded per year
n_rec_per_year = RECORDINGS_PER_MOUSE * n_mice_rec_per_year
# raw ephys
nprobes = 2
nchannels = 385
#Neuropixel 30kHz + 2kHz, 16bits, 385 channels, 2 probes
size_recording_neuropixel_hourly_Gb = 32000 * nchannels * 2 * 3600 / 1024 / 1024 / 1024 * nprobes
Tb['ephys_raw'] = size_recording_neuropixel_hourly_Gb * EPHYS_SESSION_DURATION_HOURS * n_rec_per_year * 1 / 1024
# video ephys
Tb['ephys_videos'] = SIZE_RECORDING_VIDEO_HOURLY_GB * n_rec_per_year * EPHYS_SESSION_DURATION_HOURS / 1024
Tb['ephys_alf'] = n_rec_per_year * SIZE_ALF_DATA_PER_RECORDING_SESSION_GB / 1024


# histology
histology_size_per_mouse_Gb = 30
Tb['histology'] = n_mice_rec_per_year * histology_size_per_mouse_Gb / 1024


print('\n Throughput')
print('\t', str(n_ses_per_year), ' training sessions per year')
print('\t', n_rec_per_year, ' IBL recording sessions in a year')

tot_train = sum([Tb[tb] for tb in Tb if ('training' in tb and Tb[tb] is not None)])
print('\n Training', np.round(tot_train))
print('\t', np.round(Tb['training_videos']), 'Tb of training videos')
print('\t', np.round(Tb['training_alf']), 'Tb of training ALF files')

tot_rec = sum([Tb[tb] for tb in Tb if ('ephys' in tb and Tb[tb] is not None)])
print('\n Recording:', np.round(tot_rec))
print('\t', np.round(Tb['ephys_videos']), ' Tb of recording videos')
print('\t',np.round(Tb['ephys_raw']), ' Tb of recording neuropixel data')
print('\t',np.round(Tb['ephys_alf']), ' Tb of ephys ALF data')

print('\n Histology:', np.round(Tb['histology']))


tot = sum([Tb[tb] for tb in Tb if  Tb[tb] is not None])
print('\n TOTAL', np.round(tot))


# Training
# [18270 28188]  training sessions per year
# [35.68359375 55.0546875 ] Tb of training videos
#  Recording
# [ 520. 1404.]  IBL recording sessions in a year
# [ 6.09375 32.90625]  Tb of recording videos
# [ 83.90285075 226.53769702]  Tb of recording neuropixel data
