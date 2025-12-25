# predict/chronos_model.py
import torch
#from chronos import BaseChronosPipeline
from chronos import Chronos2Pipeline



def load_chronos_model(model_path="./chronos-2"):
   pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map="cpu")
   return pipeline
