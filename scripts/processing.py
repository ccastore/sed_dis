import onnxruntime as ort
import numpy as np
import librosa as lb
import datetime
import time
import psutil
import os


def load_audio(audio,sr):
  audio,sr=lb.load(audio,sr=sr)
  return audio

def split_audio(audio,sr):
  audios=[]
  for j in np.arange(0,len(audio),10*sr):
    audio_aux=audio[j:j+(10*sr)]
    audio_aux=lb.util.normalize(audio_aux)
    if len(audio_aux) >= 3*sr:
      audios.append(audio_aux)
  return audios

def wave_to_mel(audio):
  X = np.abs(lb.stft(audio, n_fft=2048, hop_length=256, win_length=2048, window='hamming', center=True))
  mel = lb.feature.melspectrogram(sr=16000, S=X, n_fft=2048, hop_length=256, power=1.0,
                                        n_mels=128, fmin=0, fmax=8000, htk=True, norm=None)

  mel = lb.core.amplitude_to_db(mel)
  mel = np.clip(mel,-50, 80)
  mel =(mel-np.min(mel,axis=(0,1),keepdims=True))/(np.max(mel,axis=(0,1),keepdims=True)-np.min(mel,axis=(0,1),keepdims=True))
  return mel

def mel_to_model(mel,pad, yolo=True):
  mel=np.pad(mel,[(pad[0], ), (pad[1], )], mode='constant',constant_values=(0.447058824,0.447058824))
  mel= np.expand_dims(mel,0)
  
  if yolo:
    mel= np.concatenate([mel,mel,mel],axis=0)
    mel= np.expand_dims(mel,0)
  return mel

def intersection(box1,box2):
    box1_x1,box1_x2 = box1[:2]
    box2_x1,box2_x2 = box2[:2]
    x1 = max(box1_x1,box2_x1)
    x2 = min(box1_x2,box2_x2)
    return (x2-x1) 

def union(box1,box2):
    box1_x1,box1_x2 = box1[:2]
    box2_x1,box2_x2 = box2[:2]
    box1_area = (box1_x2-box1_x1)
    box2_area = (box2_x2-box2_x1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def find_changes(array):
    np_array = np.array(array)
    inicio=np.array([0])
    np_array = np.concatenate([inicio,np_array])
    np_array = np.concatenate([np_array,inicio])
    zero_to_one = np.where(np_array[:-1] < np_array[1:])[0]
    one_to_zero = np.where(np_array[:-1] > np_array[1:])[0]
    return np.concatenate([(zero_to_one/15.6).reshape(-1,1), (one_to_zero/15.6).reshape(-1,1)],axis=-1)
    
def predict(audio,ort_sess, model_type,th):
  times=[]
  times.append(("mel",datetime.datetime.now()))
  mel=wave_to_mel(audio)
  #YOLO
  if model_type=="YOLO":
    times.append(("pre",datetime.datetime.now()))
    input_sample=mel_to_model(mel,[0,7])
    times.append(("inference",datetime.datetime.now()))
    outputs = ort_sess.run(None, {'images': input_sample})
    times.append(("post",datetime.datetime.now()))
    outputs = outputs[0].transpose()[:,:,0]
    xc,w = outputs[:,0],outputs[:,2]
    cls = np.argmax(outputs[:,4:],axis=-1)
    probs= np.max(outputs[:,4:],axis=-1)
    x1 = np.clip((xc-w/2-7)*10/(640-(2*7)),0,10)
    x2 = np.clip((xc+w/2-7)*10/(640-(2*7)),0,10)
    boxes=np.array((x1,x2,cls,probs)).transpose()
    boxes=boxes[probs>th]
    boxes=boxes[np.argsort(boxes[:,-1],)][::-1]
    result = []
    while len(boxes)>0:
      result.append(boxes[0])
      boxes = [box for box in boxes if iou(box,boxes[0])<0.7]
    return result, times

  #CRNN
  elif model_type=="CRNN":
    times.append(("pre",datetime.datetime.now()))
    input_sample=mel_to_model(mel,[0,0],False)

    times.append(("inference",datetime.datetime.now()))
    outputs = ort_sess.run(None, {'input': input_sample})[0][0]
    
    times.append(("post",datetime.datetime.now()))
    result=[]
    for ind,o in enumerate(outputs):
      o_prob=o
      o = o>th
      f=find_changes(o)
      if len(f)>0:
        for i in f:
          result.append(np.array([i[0],i[1],ind,o_prob[o].mean()]))
    return result, times
      

