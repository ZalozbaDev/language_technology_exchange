from audiomentations import Compose, BandPassFilter, SomeOf, OneOf, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift, Shift, AddBackgroundNoise, PolarityInversion, RoomSimulator, ApplyImpulseResponse
import numpy as np
import sys
from scipy.io import wavfile
import glob, os
import random

random.seed(42)  # Set the seed to any integer

def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar.

    Args:
        iteration (int):   current iteration - required
        total     (int):   total iterations - required
        prefix    (str):   prefix string - optional
        suffix    (str):   suffix string - optional
        decimals  (int):   positive number of decimals in percent complete - optional
        length    (int):   character length of bar - optional
        fill      (str):   bar fill character - optional
        print_end (str):   end character (e.g. "\r", "\r\n") - optional

    Returns:
        Nothing.

    Raises:
        Nothing.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar_var = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar_var}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


augument = SomeOf((1,1), [
#augument = OneOf([
    #AddGaussianNoise(min_amplitude=0.011, max_amplitude=0.115, p=1.0),
    AddGaussianSNR(min_snr_db=5.0,max_snr_db=40.0,p=1.0),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
    #PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.05, max_fraction=0.05, p=1.0),
    AddBackgroundNoise(sounds_path="scripts/noise", min_snr_in_db=5.0, max_snr_in_db=40.0, noise_transform=PolarityInversion(),p=1.0),
    #BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0),
    #BitCrush(min_bit_depth=5, max_bit_depth=14, p=1.0),
    #ApplyImpulseResponse(ir_path="common/scripts/ir", p=1.0)
    RoomSimulator(p=1.0,
                            min_size_x=19,
                            max_size_x=23.0,
                            min_size_y=25,
                            max_size_y=50.0,
                            min_size_z=6,
                            max_size_z=12.0,
                            max_order=12,
                            min_absorption_value=0.01,
                            max_absorption_value=0.05,
                            leave_length_unchanged=True),])


'''
            min_target_rt60=1.5,
            min_source_x=0.5,
            min_source_y=0.5,
            min_source_z=1.8,
            max_source_x=0.5,
            max_source_y=0.5,
            max_source_z=1.8,
            min_mic_distance=0.1,
            max_mic_distance=0.3,
            max_order=12,
            p=1.0,
'''

'''Main loop'''
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} inputfolder outputfolder')
        raise SystemExit()

    '''
        *   inputfolder    - folder where all wav files will be augmented
        *   outfile        - folder for the augmented files
    '''
    files = glob.glob(sys.argv[1]+'/**/*.wav', recursive=True)
    clen = len(files)
    print(sys.argv[1], clen)
    cnt=0
    print_progress_bar(0, clen, prefix='Parsing:', suffix='Complete', length=50)

    for f in files:
        samplerate, data = wavfile.read(f)
        augmented_samples = random.uniform(0.35, 0.95)*augument(samples=data.astype(np.float32), sample_rate=16000)
        ofile=f.replace(sys.argv[1],sys.argv[2])
        #print(ofile)
        os.makedirs(os.path.dirname(ofile), exist_ok=True)
        wavfile.write(ofile, 16000, augmented_samples.astype(np.int16))
        cnt+=1
        print_progress_bar(cnt, clen, prefix='Parsing:', suffix='Complete', length=50)

    print('end')