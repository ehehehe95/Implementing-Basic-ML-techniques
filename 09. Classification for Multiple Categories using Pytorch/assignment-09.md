# Classification for Multiple Categories using Pytorch

Build a classifier for the digit classification task with 10 classes on the MNIST dataset

#### 1. Data

- apply normalization

``` python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),  # mean value = 0.1307, standard deviation value = 0.3081
])
```

- load the MNIST dataset

``` python
data_path = './MNIST'

training_set = datasets.MNIST(root = data_path, train= True, download=True, transform= transform)
testing_set = datasets.MNIST(root = data_path, train= False, download=True, transform= transform)
```

#### 2. Model

- design a neural network that consists of three `fully connected layers` with an activation function of `Sigmoid`
- the activation function for the output layer is `LogSoftmax`

``` python
class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        
        # construct layers for a neural network
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=20*20),
            nn.Sigmoid(),
        ) 
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=20*20, out_features=10*10),
            nn.Sigmoid(),
        ) 
        self.classifier3 = nn.Sequential(
            nn.Linear(in_features=10*10, out_features=10),
            nn.LogSoftmax(dim=1),
        ) 
        
        
    def forward(self, inputs):                 # [batchSize, 1, 28, 28]
        x = inputs.view(inputs.size(0), -1)    # [batchSize, 28*28]
        x = self.classifier1(x)                # [batchSize, 20*20]
        x = self.classifier2(x)                # [batchSize, 10*10]
        out = self.classifier3(x)              # [batchSize, 10]
        
        return out
```

#### 3. Loss function

- [the log of softmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)
- [the negative log likelihood loss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)

``` python
criterion = nn.NLLLoss()
```

#### 4. Optimization

- use a stochastic gradient descent algorithm with different mini-batch sizes of 32, 64, 128
- use a constant learning rate for all the mini-batch sizes
- do not use any regularization algorithm such as `dropout` or `weight decay`
- compute the average loss and the average accuracy for all the mini-batches within each epoch

``` python
classifier = classification().to(device)
optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate_value)
```

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

##### 1. Plot the training and testing losses with a batch size of 32 [4pt]

<img src="loss-batchsize-32.png"  width="600">

##### 2. Plot the training and testing accuracies with a batch size of 32 [4pt]

<img src="accuracy-batchsize-32.png"  width="600">

##### 3. Plot the training and testing losses with a batch size of 64 [4pt]

<img src="loss-batchsize-64.png"  width="600">

##### 4. Plot the training and testing accuracies with a batch size of 64 [4pt]

<img src="accuracy-batchsize-64.png"  width="600">

##### 5. Plot the training and testing losses with a batch size of 128 [4pt]

<img src="loss-batchsize-128.png"  width="600">

##### 6. Plot the training and testing accuracies with a batch size of 128 [4pt]

<img src="accuracy-batchsize-128.png"  width="600">


##### 7. Print the loss at convergence with different mini-batch sizes [3pt]

| mini-batch size   | 32         | 64         | 128        |
| ----------------- | ---------- | ---------- | ---------- |
| training loss     | 0.07       | 0.14       | 0.24       |
| testing loss      | 0.18       | 0.60       | 1.87       |

##### 8. Print the accuracy at convergence with different mini-batch sizes [3pt]

| mini-batch size   | 32         | 64         | 128        |
| ----------------- | ---------- | ---------- | ---------- |
| training accuracy | 0.98       | 0.96       | 0.93       |
| testing accuracy  | 0.97       | 0.96       | 0.93       |