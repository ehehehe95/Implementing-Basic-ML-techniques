# Convolutional Neural Network for the classification task on MNIST 

Design a neural network that consists of a sequence of convolutional layers for characterizing features and a sequence of fully connected layers for classifying the characteristic features into categories.

#### 1. Data

- you can use any data normalisation method
- one example of the data normalisation is whitenning as given by:

``` python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),  # mean value = 0.1307, standard deviation value = 0.3081
])
```

- load the MNIST dataset
- use the original `training` dataset for `testing` your model
- use the original `testing` dataset for `training` your model 

``` python
data_path = './MNIST'

data_test   = datasets.MNIST(root = data_path, train= True, download=True, transform= transform)
data_train  = datasets.MNIST(root = data_path, train= False, download=True, transform= transform)
```

- Note that the number of your `training` data must be 10,000
- Note that the number of your `testing` data must be 60,000

``` python
print("the number of your training data (must be 10,000) = ", data_train.__len__())
print("hte number of your testing data (must be 60,000) = ", data_test.__len__())
```

#### 2. Model

- design a neural network architecture with a combination of convolutional layers and fully connected layers
- use any number of feature layers (convolutional layers)
- use any size of convolutional kernel_size
- use any number of classification layers (fully connected layers)
- use any dimension of classification layers
- use any type of activation functions
- one example model of the convolutional neural network is as follows:

``` python
import torch.nn as nn
import torch.nn.functional as F
import math

class MyModel(nn.Module):

    def __init__(self, num_classes=10, size_kernel=5):

        super(MyModel, self).__init__()

        # *********************************************************************
        # input parameter
        #
        # data size:
        #   mnist   : 28 * 28
        # *********************************************************************
        self.number_class   = num_classes
        self.size_kernel    = size_kernel        
        
        # *********************************************************************
        # feature layer
        # *********************************************************************
        self.conv1          = nn.Conv2d(1, 20, kernel_size=size_kernel, stride=1, padding=int((size_kernel-1)/2), bias=True)
        self.conv2          = nn.Conv2d(20, 50, kernel_size=size_kernel, stride=1, padding=int((size_kernel-1)/2), bias=True)

        self.conv_layer1    = nn.Sequential(self.conv1, nn.MaxPool2d(kernel_size=2), nn.ReLU(True))
        self.conv_layer2    = nn.Sequential(self.conv2, nn.MaxPool2d(kernel_size=2), nn.ReLU(True))

        self.feature        = nn.Sequential(self.conv_layer1, self.conv_layer2)
        
        # *********************************************************************
        # classifier layer
        # *********************************************************************
        self.fc1        = nn.Linear(50*7*7, 50, bias=True)
        self.fc2        = nn.Linear(50, num_classes, bias=True)

        self.fc_layer1  = nn.Sequential(self.fc1, nn.ReLU(True))
        self.fc_layer2  = nn.Sequential(self.fc2, nn.ReLU(True))

        self.classifier = nn.Sequential(self.fc_layer1, self.fc_layer2)
        
        self._initialize_weight()        
        
    def _initialize_weight(self):

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                
                nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))

                if m.bias is not None:

                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))
                
                if m.bias is not None:

                    m.bias.data.zero_()

    def forward(self, x):

        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
```

#### 3. Loss function

- use any type of loss function
- design the output of the output layer considering your loss function 

#### 4. Optimization

- use any stochastic gradient descent algorithm for the optimization
- use any size of the mini-batch 
- use any optimization algorithm (for example, Momentum, AdaGrad, RMSProp, Adam)
- use any regularization algorithm (for example, Dropout, Weight Decay)
- use any annealing scheme for the learning rate (for example, constant, decay, staircase)

## Submission

#### Github history [1pt]

- Use the `git` commands `commit` and `push` at your `github` account for the notebook 
- Lease a history for the development of each meaningful block of the codes at `github`
- Make at least 10 `commit` 
- Save and submit the history page in PDF format

#### Python notebook

- the submission should be made in PDF format
- the notebook should consists of the following two parts:
    - main codes, comments and results (which will not be considered in the evaluation)
    - visualization codes and outputs (opon which the evaluation will be made based)

#### [output]

##### 1. Plot the training and testing losses over epochs [2pt]

##### 2. Plot the training and testing accuracies over epochs [2pt]

##### 3. Print the final training and testing losses at convergence [2pt]

- `*** NOTE ***` The values should be presented up to 5 decimal places (소수점 5째 자리까지 표기하시오)
 
| loss              |            |
| ----------------- | ---------- |
| training          | 0.12345    | 
| testing           | 0.12345    | 

##### 4. Print the final training and testing accuracies at convergence [20pt]

- `*** NOTE ***` The values should be presented up to 5 decimal places (소수점 5째 자리까지 표기하시오)

| accuracy          |            |
| ----------------- | ---------- |
| training          | 0.98765    | 
| testing           | 0.97650    | 

- The score will be determined based on the testing accuracy at convergence
    - top 1 : 20pt
    - top 2 - 3 : 18pt
    - top 4 - 6 : 16pt
    - top 7 - 10 : 14pt
    - top 11 - 15 : 12pt
    - top 16 - 21 : 10pt 
    - top 22 - 28 : 9pt
    - top 28 - 35 : 8pt
    - the rest : 7pt

##### 5. Print the testing accuracies within the last 10 epochs [5pt]

- `*** NOTE ***` The values should be presented up to 5 decimal places (소수점 5째 자리까지 표기하시오)

[epoch = 0991] 0.97641  
[epoch = 0992] 0.97642  
[epoch = 0993] 0.97643  
[epoch = 0994] 0.97644  
[epoch = 0995] 0.97645  
[epoch = 0996] 0.97646  
[epoch = 0997] 0.97647  
[epoch = 0998] 0.97648  
[epoch = 0999] 0.97649  
[epoch = 1000] 0.97650  