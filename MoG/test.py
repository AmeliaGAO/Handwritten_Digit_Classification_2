from kmeans import *
import sys
import matplotlib.pyplot as plt
def calculate1Percent(value):
  """ Calculate the percentage of 1 in value """
  numof1 = 0;
  for i in value:
    if(i==1):
      numof1 = numof1+1
  return numof1*1.0/value.shape[0]

def main():
  pd1=np.zeros(3)
  pd2=np.zeros(3)
  pd1[0]=1
  pd1[1]=2
  pd1[2]=3
  pd2[0]=1
  pd2[1]=3
  pd2[2]=1
  print(pd1.shape)
  value = np.zeros(pd1.shape)
  value[pd1>=pd2]=1
  value[pd1<pd2]=2
  print(value)
  print(calculate1Percent(value))

if __name__ == '__main__':
  main()
