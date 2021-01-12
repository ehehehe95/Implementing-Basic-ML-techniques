# Optimal Selection of the hyper-parameters associated with the classification on MNIST 

Choose an optimal set of hyper-parameters and design a neural network for the classification of MNIST dataset

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

- design a neural network architecture with three layers (input layer, one hidden layer and output layer)
- the input dimension of the input layer should be 784 (28 * 28)
- the output dimension of the output layer should be 10 (class of digits)
- all the layers should be `fully connected layers`
- use any type of activation functions

``` python
class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        
        # construct layers for a neural network
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=dim_layer1_out),
            nn.activation_layer1,
        ) 
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=dim_layer2_in, out_features=dim_layer2_out),
            nn.activation_layer2,
        ) 
        self.classifier3 = nn.Sequential(
            nn.Linear(in_features=dim_layer3_in, out_features=10),
            nn.activation_layer3,
        ) 
    
    def forward(self, inputs):                 # [batchSize, 1, 28, 28]
        x = inputs.view(inputs.size(0), -1)    # [batchSize, 28*28]
        x = self.classifier1(x)                # [batchSize, 20*20]
        x = self.classifier2(x)                # [batchSize, 10*10]
        out = self.classifier3(x)              # [batchSize, 10]
        
        return out
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

| loss              |            |
| ----------------- | ---------- |
| training          | 0.1        | 
| testing           | 0.2        | 

##### 4. Print the final training and testing accuracies at convergence [20pt]

| accuracy          |            |
| ----------------- | ---------- |
| training          | 0.99       | 
| testing           | 0.98       | 

- The score will be determined based on the testing accuracy
    - top 1 : 20pt
    - top 2 - 3 : 18pt
    - top 4 - 6 : 16pt
    - top 7 - 10 : 14pt
    - top 11 - 15 : 12pt
    - top 16 - 21 : 10pt 
    - top 22 - 28 : 9pt
    - top 28 - 35 : 8pt
    - the rest : 7pt