import sys, os
os.chdir(os.path.dirname(__file__)) # go to current file directory

import numpy as np
import matplotlib.pyplot as plt

"Sound processing"
import sounddevice as sd
import soundfile as sf
import random
import librosa
from scipy import signal

# ----------------------------------------------------------------------------------
"""
Synthesis of the functions and classes in :
- resample : resample signal x to frequency fs2
- specgram : get a spectrogram from a 1D audio signal
- get_Hz2mel : get the hz2mel matrix transformation
- melspecgram : get a melspectrogram from a 1D audio signal

- AudioUtil : util functions to process an audio signal
- SoundDS : Create a dataset object
"""
# ----------------------------------------------------------------------------------

def resample(x, fs=44100, fs2=11025):
    """   Resample signal x to frequency fs2.
    
    Inputs
      x:  [ndarray, size:N, input signal]
      fs: [float, initial sampling frequency]
      fs2: [float, target sampling frequency]

    Outputs
      y:   [ndarray, size:M*N, resample signal] 

    """
    ### TO COMPLETE

    return x

def specgram(x, Nft=512, fs=44100, fs2=11025):
    """ Get a spectrogram from a 1D audio signal
    
    Inputs
      y:  [ndarray, size:N, input signal]
      Nft:  [int, number of frequencies for FFT]

    Outputs
      stft:   [ndarray, size:(Nft,N/Nft), spectrogram] 

    """

    ### TO COMPLETE

    return stft

def get_hz2mel(fs2=11025, Nft=512, Nmel=20):
    """ Get the hz2mel matrix transformation
    Inputs
      Nmel: [int, number of mels]

    Outputs:
      mels: [ndarray, size: (Nmel,Nft), hz2mel matrix]
    """
    mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
    mels = mels[:,:-1]
    mels = mels/np.max(mels)

    return mels

def melspecgram(x, Nmel=20, Nft=512, fs=44100, fs2=11025, M=4):
    """ Get a melspectrogram from a 1D audio signal
    
    Inputs
      x:  [ndarray, size:N, input signal]
      Nmel:  [int, number of mel coefficients]
      Nft:  [int, number of frequencies for FFT]
      fs:  [float, sampling frequency]
      fs2:  [float, new sampling frequency]

    Outputs
      melspec:   [ndarray, size:(Nmel, N/Nft), melspectrogram] 

    """
    ### TO COMPLETE

    return melspec




class AudioUtil():
  """
    Define a new class with util functions to process an audio signal.
  """
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = sf.read(audio_file)
    if (sig.ndim>1):
        sig = sig[:,0]
    return (sig, sr)

  # ----------------------------
  # Play an audio file.
  # ----------------------------
  @staticmethod
  def play(audio):
      sig, sr = audio
      sd.play(sig, sr)

  # ----------------------------
  # Resample to target sampling frequency
  # ----------------------------
  @staticmethod
  def resample(aud, newsr=11025):
    sig, sr = aud
    resig = resample(sig, sr, newsr)

    return ((resig, newsr))

  # ----------------------------
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    sig_len = len(sig)
    max_len = int(sr * max_ms/1000)

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = np.zeros(pad_begin_len)
      pad_end = np.zeros(pad_end_len)

      # sig = np.append([pad_begin, sig, pad_end])
      sig = np.concatenate((pad_begin, sig, pad_end))
      
    return (sig, sr)

  # ----------------------------
  # Shifts the signal to the left or right by some percent. Values at the end
  # are 'wrapped around' to the start of the transformed signal.
  # ----------------------------
  @staticmethod
  def time_shift(aud, shift_limit=0.4):
    sig,sr = aud
    sig_len = len(sig)
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (np.roll(sig, shift_amt), sr)

  # ----------------------------
  # Augment the audio signal by scaling it by a random factor. 
  # ----------------------------
  @staticmethod
  def scaling(aud, scaling_limit=5):
    sig,sr = aud

    ### TO COMPLETE

    return aud

  # ----------------------------
  # Augment the audio signal by adding gaussian noise. 
  # ----------------------------
  @staticmethod
  def add_noise(aud, sigma=0.05):
    sig,sr = aud
    ### TO COMPLETE

    return aud

  # ----------------------------
  # Augment the audio signal by adding another one in background. 
  # ----------------------------
  @staticmethod
  def add_bg(aud, allpath, num_sources=1, max_ms=5000, amplitude_limit=0.1):
    """
    Adds up sounds uniformly chosen at random to aud.

    Inputs
    aud:  [2-tuple, (1D audio signal, sampling frequency)]
    allpath: [2D str list, size:(n_classes, n_sounds), contains the paths to each sound]
    num_sources : [int, number of sounds added in background]
    max_ms : [float, duration of aud in milliseconds, necessary to pad_trunc]
    amplitude_limit: [float, maximum ratio between amplitudes of additional sounds and sound of interest]

    Outputs
      aud_bg:  [2-tuple, (1D audio signal with additional sounds in background, sampling frequency)] 
    """  

    sig,sr = aud

    ### TO COMPLETE

    return aud

  # ----------------------------
  # Generate a Spectrogram
  # ----------------------------
  @staticmethod
  def melspectrogram(aud, Nmel=20, Nft=512, fs2=11025):
    sig,sr = aud
    return melspecgram(sig, Nft=Nft, Nmel=Nmel, fs=sr, fs2=fs2)

  # ----------------------------
  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
  # overfitting and to help the model generalise better. The masked sections are
  # replaced with the mean value.
  # ----------------------------
  @staticmethod
  def spectro_aug_timefreq_masking(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    Nmel, n_steps = spec.shape
    mask_value = np.mean(spec)
    aug_spec = np.copy(spec) # avoids modifying spec

    freq_mask_param = max_mask_pct * Nmel
    for _ in range(n_freq_masks):
      height = int(np.round(random.random()*freq_mask_param))
      pos_f = np.random.randint(Nmel-height)
      aug_spec[pos_f:pos_f+height,:] = mask_value

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      width = int(np.round(random.random()*time_mask_param))
      pos_t = np.random.randint(n_steps-width)
      aug_spec[:, pos_t:pos_t+width] = mask_value

    return aug_spec

#_____________________________________________________________________________________________________________

class SoundDS():
  """
    Sound Dataset
  """
  def __init__(self, class_ids, data_path=None, allpath_mat=None, melspec=None, data_aug=[None]):
    self.class_ids = class_ids
    self.melspec = melspec
    self.data_path = data_path
    self.allpath_mat = allpath_mat
    self.duration = 5000 # ms
    self.sr = 11025
    self.shift_pct = 0.4 # percentage of total
    self.data_aug = data_aug
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return np.size(self.data_path)
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):

    # Get the Class ID
    class_id = self.class_ids[idx]

    if (self.melspec is not None):
      sgram_crop = self.melspec[:, :10]
    else:
      audio_file = self.data_path[idx]
      aud = AudioUtil.open(audio_file)
      aud = AudioUtil.resample(aud, self.sr)
      aud = AudioUtil.pad_trunc(aud, self.duration)
      if (self.data_aug[0] is not None):
        if ('add_bg' in self.data_aug):
          aud = AudioUtil.add_bg(aud, self.allpath_mat, num_sources=1, max_ms=self.duration, amplitude_limit=0.1)
        if ('noise' in self.data_aug):
          aud = AudioUtil.add_noise(aud, sigma=0.05)
        if ('scaling' in self.data_aug):
          aud = AudioUtil.scaling(aud, scaling_limit=5)

      aud = AudioUtil.time_shift(aud, self.shift_pct)
      sgram = AudioUtil.melspectrogram(aud, Nmel=20, Nft=512)
      if ('aug_sgram' in self.data_aug):
        sgram = AudioUtil.spectro_aug_timefreq_masking(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

      sgram_crop = sgram[:, :10]
    return sgram_crop, class_id
