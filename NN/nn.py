from util import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def TrainNN(num_hiddens, eps, momentum, num_epochs):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
   
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  W = np.zeros((W1.shape[0], 4))
  W[:,0] = W1[:,0]
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error = []
  valid_error = []
  num_train_cases = inputs_train.shape[1]
  lowest_valid_err = 1
  convergence_epoch = 0
  W1_1 = np.zeros(W1.shape)
  W1_2 = np.zeros(W1.shape)
  for epoch in xrange(num_epochs):
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    # Compute cross entropy
    #train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))
    train_CE = np.mean((np.abs(np.subtract(target_train, prediction))*2).astype(int))

    # Compute deriv
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    # Get W1_1 and W1_2 to visualize the input to hidden weights change
    if(epoch == 10):
      W[:,1] = W1[:,0]
    if(epoch == 20):
      W[:,2] = W1[:,0]

    valid_CE = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)

    #update the lowest_valid_err
    if(valid_CE < lowest_valid_err):
      lowest_valid_err = valid_CE
      convergence_epoch = epoch

    train_error.append(train_CE)
    valid_error.append(valid_CE)
    # sys.stdout.write('\rStep %d Train CE %.5f Validation CE %.5f' % (epoch, train_CE, valid_CE))
    # sys.stdout.flush()
    # if (epoch % 100 == 0):
    #   sys.stdout.write('\n')

  #sys.stdout.write('\n')
  final_train_error = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_error = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_error = Evaluate(inputs_test, target_test, W1, W2, b1, b2)
  W[:,3] = W1[:,0]
  #print 'Error: Train %.5f Validation %.5f Test %.5f' % (final_train_error, final_valid_error, final_test_error)
  return W1, W2, b1, b2, train_error, valid_error,lowest_valid_err,convergence_epoch, final_test_error,W

def Evaluate(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  #CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)) # Calculate error with cross entropy
  CE = np.mean((np.abs(np.subtract(target, prediction))*2).astype(int)) # Calculate error with number of incorrectly predicted value
  return CE

def DisplayErrorPlot(train_error, valid_error):
  """Plot the error rate of training set and validation set against epoch."""
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Train')
  plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel('Error Rate')
  plt.legend()
  plt.draw()
  raw_input('Press Enter to exit.')

def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error, 'valid_error' : valid_error}
  print 'Writing model to %s' % modelfile
  np.savez(modelfile, **model)

def LoadModel(modelfile):
  """Loads model from numpy file."""
  model = np.load(modelfile)
  return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

def plotImage(data):
  """Show the cluster centers as images."""
  plt.figure(1)
  plt.clf()
  for i in xrange(data.shape[1]):
    plt.subplot(1, 4, i+1)
    plt.imshow(data[:, i].reshape(16, 16).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')

def main():
  num_hiddens = 30
  eps = 0.5
  momentum = 0.9
  num_epochs = 1000

  W1, W2, b1, b2, train_error, valid_error, lowest_valid_err, convergence_epoch,final_test_error, W = TrainNN(num_hiddens, eps, momentum, num_epochs)
  # W contains the 4 frames of the hidden layer weights
  plotImage(W)

  # print lowest_valid_err, final_test_error
  # DisplayErrorPlot(train_error, valid_error)

  # Compare the cross entropy and the convergence point when parameters change by plotting them out 
  #epsArr = np.array([0.01,0.1,0.2,0.5])
  #mArr = np.array([0.0, 0.5, 0.9])
  # hiddenArr = np.array([2,5,10,30,100])
  # N = hiddenArr.shape[0]
  # validErr = np.zeros(N)
  # convergeEpo = np.zeros(N)
  # for i in xrange(N):
  #   W1, W2, b1, b2, train_error, valid_error, lowest_valid_err, convergence_epoch = TrainNN(hiddenArr[i], eps, momentum, num_epochs)
  #   validErr[i] = lowest_valid_err
  #   convergeEpo[i] = convergence_epoch

  # plt.clf()
  # x = hiddenArr
  # plt.plot(x,validErr)
  # plt.xlabel('Number of Hidden Units')
  # plt.ylabel('Validation Cross Entropy')
  # plt.draw()
  # raw_input('Press Enter to exit.')

  # plt.clf()
  # x = hiddenArr
  # plt.plot(x,convergeEpo)
  # plt.xlabel('Number of Hidden Units')
  # plt.ylabel('Convergence Epochs')
  # plt.draw()
  # raw_input('Press Enter to exit.')

  # If you wish to save the model for future use :
  # outputfile = 'model.npz'
  # SaveModel(outputfile, W1, W2, b1, b2, train_error, valid_error)

if __name__ == '__main__':
  main()
