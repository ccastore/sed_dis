import onnxruntime as ort
import datetime
import time
import numpy as np
from onnxruntime import GraphOptimizationLevel
import os
from scripts.processing import predict, load_audio
import psutil
import tqdm

model_type="YOLO" #select model YOLO / CRNN
providers=['CPUExecutionProvider']
quantize="_mixed" #_orinigal, _pruned_01, _pruned_001, quantized_int8, quantized_int8_d
reduction="True"

if reduction:
  opt_options = ort.SessionOptions()
  opt_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED


if model_type=="YOLO":
  ort_sess = ort.InferenceSession('models/YOLOv5n'+quantize+'.onnx', opt_options, providers=providers)
  th=0.2 #Threshold of the confidence of the prediction
elif model_type=="CRNN":
  ort_sess = ort.InferenceSession('models/CRNN'+quantize+'.onnx', opt_options,providers=providers)
  th=0.5 #Threshold of the confidence of the prediction

classes=["Horn","Siren","Pets","Physiological","Speech","RingTone","Vibrating","Notifications","Cry"] #List of predictions
sr=16000

files=os.listdir("samples")
files=np.random.choice(files,180)

times=[]
for file in tqdm.tqdm(files):
  inicio=datetime.datetime.now()
  dir_audio="samples/"+file
  audio=load_audio(dir_audio,sr) 
  fin=inicio=datetime.datetime.now()
  delay=10-(fin-inicio).total_seconds()
  time.sleep(delay)
  fin=datetime.datetime.now() 

  p_complete,t=predict(audio, ort_sess,model_type,th)
  process = psutil.Process(os.getpid())
  memory = process.memory_info().rss / float(2 ** 20)
  cpu_percent = psutil.cpu_percent(interval=None)
  cpu_percent =f"{cpu_percent:.2f}"
  temp_info = psutil.sensors_temperatures()
  if "coretemp" in temp_info:
    core_temps = temp_info["coretemp"]
    avg_temp = sum(sensor.current for sensor in core_temps) / len(core_temps)
    temp_info=f"{avg_temp:.2f}"
  else:
    temp_info=-1

  mel_time=float(str((t[0][2]-t[0][1])).split(":")[-1])
  preprocessing_time=float(str((t[1][2]-t[1][1])).split(":")[-1])
  inference_time=float(str((t[2][2]-t[2][1])).split(":")[-1])
  postprocessing_time=float(str((t[3][2]-t[3][1])).split(":")[-1])
  times.append([mel_time,preprocessing_time,inference_time,postprocessing_time, memory,cpu_percent,temp_info])

np.save(model_type+quantize+"_times.npy",np.array(times))