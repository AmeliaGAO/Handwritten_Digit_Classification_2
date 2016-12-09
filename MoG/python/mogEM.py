from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary=0):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1) #calculate the mean of the same row, reshape if to (length of the array, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst) #initialize by random
  mu = KMeans(x, K, 5) # initialize by ranning kmeans for 5 iterations
  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    # print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # # Plot log prob of data
    # plt.figure(1);
    # plt.clf()
    # plt.plot(np.arange(i), logProbX[:i], 'r-')
    # plt.title('Log-probability of data versus # iterations of EM')
    # plt.xlabel('Iterations of EM')
    # plt.ylabel('log P(D)');
    # plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def plotImage(data):
  """Show the cluster centers as images."""
  plt.figure(1)
  plt.clf()
  for i in xrange(data.shape[1]):
    plt.subplot(1, data.shape[1], i+1)
    plt.imshow(data[:, i].reshape(16, 16).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')

def q2():
  iters = 10
  minVary = 0.01
  K = 2
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

  p2,mu2,vary2,logProbX2 = mogEM(train2, K, iters, minVary)
  print(p2)
  raw_input('Press Enter to continue.')
  #plotImage(mu2)
  #plotImage(vary2)

  p3,mu3,vary3,logProbX3 = mogEM(train3, K, iters, minVary)
  print(p3)
  raw_input('Press Enter to continue.')
  #plotImage(mu3)
  #plotImage(vary3)

def q3():
  iters = 20
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  p,mu,vary,logProbX = mogEM(inputs_train, 20, iters, minVary)
  # plotImage(mu)
  # plotImage(vary)

  raw_input('Press Enter to continue.')

def classifyX(p2,mu2,vary2,x,p3,mu3,vary3):
  """ 
  Calculate the p(x|d=1) and p(x|d=2). 
  Use bayes rule to get p(d=1|x) and p(d=2|x). 
  Assign the value to 1 if p(d=1|x) > p(d=2|x) else assign the value to 2
  Return value
  """
  pxGivenD1 = np.exp(mogLogProb(p2,mu2,vary2,x))
  pxGivenD2 = np.exp(mogLogProb(p3,mu3,vary3,x))
  px = pxGivenD1*0.5 + pxGivenD2*0.5
  pd1 = pxGivenD1*0.5/px
  pd2 = pxGivenD2*0.5/px
  value = np.zeros(pd1.shape)
  value[pd1>pd2] = 1
  value[pd1<=pd2] =2;
  return value
  

def calculate1Percent(arr):
  """ Calculate the percentage of 1 in array arr. The only value in array arr are 1 and 2 """
  numof1 = 0;
  for i in arr:
    if(i==1):
      numof1 = numof1+1
  return numof1*1.0/arr.shape[0]

def calAvgErr(p2,mu2,vary2,x2,p3,mu3,vary3,x3):
  """ Calculate error rate for 2's and 3's. Get the average """
  classifiedResult2 = classifyX(p2,mu2,vary2,x2,p3,mu3,vary3)
  error2 = 1- calculate1Percent(classifiedResult2)
  classifiedResult3 = classifyX(p2,mu2,vary2,x3,p3,mu3,vary3)
  error3 = calculate1Percent(classifiedResult3)
  return (error2+error3)/2

def q4():
  iters = 10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  print(errorTrain)
  numComponents = np.array([2, 5, 15, 25])
  T = numComponents.shape[0]  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  N = inputs_train.shape[0]

  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    p2,mu2,vary2,logProbX2 = mogEM(train2, K, iters, minVary)
    
    # Train a MoG model with K components for digit 3
    p3,mu3,vary3,logProbX3 = mogEM(train3, K, iters, minVary)

    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    errorTrain[t] = calAvgErr(p2,mu2,vary2,train2,p3,mu3,vary3,train3)
    errorValidation[t] = calAvgErr(p2,mu2,vary2,valid2,p3,mu3,vary3,valid3)
    errorTest[t] = calAvgErr(p2,mu2,vary2,test2,p3,mu3,vary3,test3)
    
  print(errorTrain);  
  print(errorValidation);  
  print(errorTest);  
  # Plot the error rate
  plt.clf()
  x = numComponents
  plt.plot(x,errorTrain)
  plt.plot(x,errorValidation)
  plt.plot(x,errorTest)
  plt.legend(['Train','Validation','Test'],loc='upper right')
  
  plt.draw()
  raw_input('Press Enter to continue.')

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  K = 15;
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  p2,mu2,vary2,logProbX2 = mogEM(train2, K, iters, minVary)
  p3,mu3,vary3,logProbX3 = mogEM(train3, K, iters, minVary)

  errorTrain = calAvgErr(p2,mu2,vary2,train2,p3,mu3,vary3,train3)
  errorValidation = calAvgErr(p2,mu2,vary2,valid2,p3,mu3,vary3,valid3)
  errorTest = calAvgErr(p2,mu2,vary2,test2,p3,mu3,vary3,test3)
  print(errorTrain);  
  print(errorValidation);  
  print(errorTest);  
  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  #q2()
  #q3()
  #q4()
  q5()

