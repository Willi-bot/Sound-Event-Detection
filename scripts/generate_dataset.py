import scaper
import numpy as np
import os

# OUTPUT FOLDER
outfolder = 'soundscapes'

# SCAPER SETTINGS
fg_folder = 'source/foreground/'
bg_folder = 'source/background/'

n_soundscapes = 32000
ref_db = -40
duration = 10.0

min_events = 1
max_events = 4

min_bg = 1
max_bg = 2

event_time_dist = 'truncnorm'
event_time_mean = 5.0
event_time_std = 2.0
event_time_min = 0.0
event_time_max = 10.0

source_time_dist = 'const'
source_time= 0.0

event_duration_dist = 'uniform'
event_duration_min = 1.
event_duration_max = 5.5

snr_dist = 'uniform'
snr_min = 3
snr_max = 15

pitch_dist = 'uniform'
pitch_min = -2.0
pitch_max = 2.0

time_stretch_dist = 'uniform'
time_stretch_min = 0.9
time_stretch_max = 1.1

# set seed for randomness
seed = 42

# create a scaper that will be used below
sc = scaper.Scaper(duration, fg_folder, bg_folder, random_state=seed)
sc.protected_labels = []
sc.ref_db = ref_db

# Generate n_soundscapes soundscapes using a truncated normal distribution of start times

for n in range(n_soundscapes):

    print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))

    # reset the event specifications for foreground and background at the
    # beginning of each loop to clear all previously added events
    sc.reset_bg_event_spec()
    sc.reset_fg_event_spec()

    # add noise
    sc.add_background(label=('choose', ['white_noise']),
                      source_file=('choose', []),
                      source_time=('const', 0))

    # add background
    n_bg = np.random.randint(min_bg, max_bg+1)
    for _ in range(n_bg):
        sc.add_background(label=('choose', ['wind', 'footsteps', 'frog', 'insects', 'rain', 'crickets']),
                          source_file=('choose', []),
                          source_time=('const', 0))

    # add random number of foreground events
    n_events = np.random.randint(min_events, max_events+1)
    for _ in range(n_events):
        sc.add_event(label=('choose', []),
                     source_file=('choose', []),
                     source_time=(source_time_dist, source_time),
                     event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max),
                     event_duration=(event_duration_dist, event_duration_min, event_duration_max),
                     snr=(snr_dist, snr_min, snr_max),
                     pitch_shift=(pitch_dist, pitch_min, pitch_max),
                     time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))

    # generate
    audiofile = os.path.join(outfolder, "audio/{:d}.wav".format(n))
    jamsfile = os.path.join(outfolder, "jams/{:d}.jams".format(n))
    txtfile = os.path.join(outfolder, "metadata/{:d}.txt".format(n))

    sc.generate(audiofile, jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=False,
                reverb=0.1,
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile)