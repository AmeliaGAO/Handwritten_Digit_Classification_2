from util import *
import sys

def main():
  target = np.array([[ 0.0, 0.0, 1.0, 1.0 ],[ 0.7, 0.7, 0.2, 0.7]])
  new = np.zeros((target.shape[0],1))
  new = target[:,0]
  print new

if __name__ == '__main__':
  main()
