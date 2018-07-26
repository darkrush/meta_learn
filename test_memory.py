from memory import Memory
import numpy as np
import random

act_shape = (2,2)
ob_shape = (2,2)
memory_size = 100


def gen_trans():
  ob0 =  np.random.rand(*ob_shape)
  act =  np.random.rand(*act_shape)
  rwd =  np.random.rand(1)
  ob1 =  np.random.rand(*ob_shape)
  trm =  np.random.rand(1)
  return ob0,act,rwd,ob1,trm



memory = Memory(memory_size,act_shape,ob_shape)


for i in range(200):
  args = gen_trans()
  memory.append(*args)
  
