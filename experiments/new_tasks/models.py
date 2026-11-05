import torch
from torch import nn
class SequenceModel(nn.Module):
 def __init__(self,meta):
  super().__init__();self.family=meta['family'];self.rnn=nn.LSTM(meta['input_dim'],64,1,batch_first=True)
  self.horizon=meta.get('horizon',1);self.outputs=meta.get('output_dim',meta.get('classes'))
  self.head=nn.Linear(64,self.outputs*self.horizon)
 def forward(self,x):
  h=self.rnn(x)[0][:,-1];y=self.head(h)
  return y.reshape(len(x),self.horizon,self.outputs) if self.family=='forecasting' else y
class Denoiser(nn.Module):
 def __init__(self,d):
  super().__init__();self.layers=nn.Sequential(nn.Linear(d,64),nn.Tanh(),nn.Linear(64,16),nn.Tanh(),nn.Linear(16,64),nn.Tanh(),nn.Linear(64,d),nn.Sigmoid())
 def forward(self,x):return self.layers(x)
def make_model(meta):return Denoiser(meta['input_dim']) if meta['family']=='denoising' else SequenceModel(meta)
