"""
--------------------------------------------------------------------------
    pedestrian crop image anomaly
    H.C. Shin, creatrix@etri.re.kr, 2019.10.24
--------------------------------------------------------------------------
    Copyright (C) <2019>  <H.C. Shin, creatrix@etri.re.kr>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
--------------------------------------------------------------------------
"""
from __future__ import print_function
import argparse
import queue
import os
import sys
import numpy as np
import scipy.signal
import scipy
import math
import fractions
import librosa
from keras.optimizers import Adam
from keras.layers import Dense, Merge
from keras.models import Sequential
from package import vggish, mel_features, vggish_params

# 4채널 오디오 입력 장치에서 동작, 방수 마이크 입력 가정
num_direction = 4
num_class = 5
event_type_list = [str(i)+str(j) for i in range(num_class) for j in range(num_class) if i != j]
mic_type_list = [str(i)+str(j) for i in range(num_direction) for j in range(num_direction) if i < j]
onehot_length = num_class*num_direction + len(mic_type_list) * len(event_type_list) + 1


# 데이터 파싱을 위한 함수
def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=768, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=300,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
parser.add_argument(
    'channels', type=int, default=[1,2,3,4], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
args = parser.parse_args()

if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]
q = queue.Queue()
out = np.zeros((12288,4))
bf_out = np.zeros((4,12288))
predictions = []


# 빔포잉을 위한 함수(fd, delay)
def fd(signal,N):

    N=math.ceil(N*100)/100
    #get rational approximation
    result=fractions.Fraction(N).limit_denominator(20)
    num=result.numerator;
    den=result.denominator
    #upsample the signal and interpolate

    out1=scipy.signal.resample(signal,den*len(signal))
    #delay the signal
    out1=delay(out1,int(num))        
    #downsample the signal

    out1=scipy.signal.resample(out1,len(signal))
        
    output=out1
         
    return output

def delay(signal,N):
    if N==0:
        return signal;
        
    if N >= len(signal):
        N=N-len(signal)
    
    if N <0:
        N=N+len(signal)
   
    d=signal[len(signal)-N:len(signal)];   
    signal1=np.append(d,signal[0:len(signal)-N])
    return signal1;


# 오디오 -> mel 입력 특성으로 변환
def waveform_to_examples(data, sample_rate):
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples

dict_class = {0:'breaking',1:'scream',2:'carhorn',3:'siren',4:'engine'}



# onehot 벡터 프린트
def onehotprint(array):
    num_label = array.shape[0]
    for i in range(num_label):
        one_locate = np.where(array[i]==1)[0][0]
        if array[i][-1] == 1:
            print('normalstate')
        elif array[i].shape[0] == 141:
            num_direction = 4           
            num_class = 5
            # event type list ['01','02','03',...,'10','12',...]
            event_type_list = [str(i)+str(j) for i in range(num_class) for j in range(num_class) if i != j]
            # mic type list ['01','02','03',...'12','13',...]
            mic_type_list = [str(i)+str(j) for i in range(num_direction) for j in range(num_direction) if i < j]
            if one_locate < num_direction*num_class:
                mic_loc = one_locate // num_class
                event_type = one_locate % num_class
                print('****Single source****')
                print('Direction: {}, Event: {}'.format(mic_loc*(360//num_direction), dict_class[event_type]))
            else:
                events_type = (one_locate - num_direction*num_class ) % len(event_type_list)
                mic_type = (one_locate - num_direction*num_class ) // len(event_type_list)               
                event_type1 = int(event_type_list[events_type][0])
                event_type2 = int(event_type_list[events_type][1])
                mic_loc1 = int(mic_type_list[mic_type][0])
                mic_loc2 = int(mic_type_list[mic_type][1])
                print('****Multi source****')
                print('Direction: {}, Event: {}'.format(mic_loc1*(360//num_direction), dict_class[event_type1]))
                print('Direction: {}, Event: {}'.format(mic_loc2*(360//num_direction), dict_class[event_type2]))
        else:
            num_direction = 8           
            num_class = 5
            # event type list ['01','02','03',...,'10','12',...]
            event_type_list = [str(i)+str(j) for i in range(num_class) for j in range(num_class) if i != j]
            # mic type list ['01','02','03',...'12','13',...]
            mic_type_list = [str(i)+str(j) for i in range(num_direction) for j in range(num_direction) if i < j]
            if one_locate < num_direction*num_class:
                mic_loc = one_locate // num_class
                event_type = one_locate % num_class
                print('****Single source****')
                print('Direction: {}, Event: {}'.format(mic_loc*(360//num_direction), dict_class[event_type]))
            else:
                events_type = (one_locate - num_direction*num_class ) % len(event_type_list)
                mic_type = (one_locate - num_direction*num_class ) // len(event_type_list)
                event_type1 = int(event_type_list[events_type][0])
                event_type2 = int(event_type_list[events_type][1])
                mic_loc1 = int(mic_type_list[mic_type][0])
                mic_loc2 = int(mic_type_list[mic_type][1])
                print('****Multi source****')
                print('Direction: {}, Event: {}'.format(mic_loc1*(360//num_direction), dict_class[event_type1]))
                print('Direction: {}, Event: {}'.format(mic_loc2*(360//num_direction), dict_class[event_type2]))


# 실시간 구동을 위한 콜백 함수
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])

# 실시간 구동
def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    global bf_out
    global predictions

    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
        out = plotdata
        
        ch_a = out.T[1];
        ch_b_0 = out.T[3];
        ch_c_0 = out.T[0];

        out_0_0 = fd(ch_c_0,1.0000)-ch_a;
        out_90_0 = fd(ch_b_0,1.0000)-ch_a;
        out_270_0 = fd(ch_a,1.0000)-ch_b_0;
        out_180_0 = fd(ch_a,1.0000)-ch_c_0;

        bf_out[0]=out_0_0
        bf_out[1]=out_90_0
        bf_out[2]=out_180_0
        bf_out[3]=out_270_0

    features = np.ndarray((1,96,64,0))
    mean = np.mean(np.mean(bf_out, axis=1))
    for i in range(4):
        pad = np.lib.pad(bf_out[i],(0,16000-bf_out[i].shape[0]),'constant')
        feature = waveform_to_examples(pad,16000)        
        features = np.append(features,np.expand_dims(feature,axis=3), axis=3)

    print("#############")
    Y_pred = model.predict([np.expand_dims(features[:,:,:,0], axis=3),
        np.expand_dims(features[:,:,:,1], axis=3),
        np.expand_dims(features[:,:,:,2], axis=3),
        np.expand_dims(features[:,:,:,3], axis=3)])
    predictions.append(Y_pred)
    if len(predictions) > 5:
        predictions.pop(0)
    average_pred = np.mean(predictions,axis=0)
    y_pred = np.argmax(average_pred, axis=1)
    
    pred1 = np.zeros((1,onehot_length))
    pred1[0][y_pred[0]] = 1 
    onehotprint(pred1)     

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

try:
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    model_1 = Sequential()
    model_1.add(vggish.VGGish(include_top=False, name='_ch1'))
    model_2 = Sequential()
    model_2.add(vggish.VGGish(include_top=False, name='_ch2'))
    model_3 = Sequential()
    model_3.add(vggish.VGGish(include_top=False, name='_ch3'))
    model_4 = Sequential()
    model_4.add(vggish.VGGish(include_top=False, name='_ch4'))

    model = Sequential()
    model.add(Merge([model_1, model_2, model_3, model_4], mode='concat'))
    model.add(Dense(128))
    model.add(Dense(128))
    model.add(Dense(onehot_length, activation='softmax'))

    opt = Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("#")

    model.load_weights('C_weights_best.h5')

    length = 12288
    plotdata = np.zeros((length, len(args.channels)))
    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    

    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom='off', top='off', labelbottom='off',
                   right='off', left='off', labelleft='off')
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=6, channels=max(args.channels),
        samplerate=16000, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
