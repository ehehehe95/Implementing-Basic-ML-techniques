# Autoencoder for image denoising

Design a neural network comprising a pair of encoder and decoder for denoising of images.

#### 1. Data

- data consist of three different sets
    - training data that consist of clean images
    - testing data that consist of pairs of clean and noisy images
- you add noise to clean images and use them for training based on (clean, noisy) image pairs

``` python
train_data_path         = './ImageNet/train'
test_data_path          = './ImageNet/test'
noise_test_data_path    = './ImageNet/testNoise'

training_set            = datasets.ImageFolder(root = train_data_path, transform= transform)
testing_set             = datasets.ImageFolder(root = test_data_path, transform= transform)
noise_testing_set       = datasets.ImageFolder(root = noise_test_data_path, transform= transform)
```

- load the MNIST dataset
- use the original `training` dataset for `testing` your model
- use the original `testing` dataset for `training` your model 

``` python
trainloader     = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=6, drop_last=True)
testloader      = DataLoader(testing_set, batch_size=32, shuffle=False, num_workers=6, drop_last=False)
noiseTestloader = DataLoader(testing_set, batch_size=32, shuffle=False, num_workers=6, drop_last=False)
```

- Note that the number of your `training` data must be 10,000
- Note that the number of your `testing` data must be 10,000
- Note that the number of your `noisy testing` data must be 10,000

``` python
print("number of training data = ", training_set.__len__())
print("number of testing data = ", testing_set.__len__())
print("number of noisy testing data = ", noise_testing_set.__len__())
```

#### 2. Model

- design a neural network architecture comprising a pair of encoder and decoder
- use any number of feature layers (convolutional layers)
- use any size of convolutional kernel_size
- use any type of activation functions

##### Example of autoencoder model with skip connections (decoder is formed by resizing and convolution)

``` python
class AutoEncoder(nn.Module):           # VGG 16
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.deconv4 = nn.Sequential(  
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        bottleNeck = self.conv4(feature3)

        out = self.deconv1(bottleNeck)
        out = self.deconv2(out + feature3)
        out = self.deconv3(out + feature2)
        out = self.deconv4(out + feature1)

        return out
```

##### Example of autoencoder model without skip connections (decoder is formed by resizing and convolution)

``` python
class AutoEncoder(nn.Module):           # VGG 16
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.deconv4 = nn.Sequential(  
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        bottleNeck = self.conv4(feature3)

        out = self.deconv1(bottleNeck)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)

        return out
```

##### Example of autoencoder model without skip connections (decoder is formed by transposed convolution)

``` python
class AutoEncoder(nn.Module):           # VGG 16
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.deconv4 = nn.Sequential(  
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        bottleNeck = self.conv4(feature3)

        out = self.deconv1(bottleNeck)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)

        return out
```

#### 3. Loss function

- use mean square error between the expected clean image and the output of the network

``` python
model = AutoEncoder()
model.to(device)
mseCriterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

#### 4. Optimization

- use any stochastic gradient descent algorithm for the optimization
- use any size of the mini-batch 
- use any optimization algorithm (for example, Momentum, AdaGrad, RMSProp, Adam)
- use any regularization algorithm (for example, Dropout, Weight Decay)
- use any annealing scheme for the learning rate (for example, constant, decay, staircase)

#### 5. Quantitative evaluation of the output

- compute PSNR between the expected clean image and the output of the network

``` python
def get_psnr(img1, img2, min_value=0, max_value=1):
    
    if type(img1) == torch.Tensor:
        mse = torch.mean((img1 - img2) ** 2)
    else:
        mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100
    
    PIXEL_MAX = max_value - min_value
    
    return 10 * log10((PIXEL_MAX ** 2) / mse)
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

##### 1. Plot the training and testing losses over epochs [2pt]

##### 2. Plot the training and testing PSNR over epochs [2pt]

##### 3. Print the final testing losses at convergence [2pt]

- `*** NOTE ***` The values should be presented up to 5 decimal places (소수점 5째 자리까지 표기하시오)
 
| loss              |            |
| ----------------- | ---------- |
| training          | 0.12345    | 
| testing           | 0.12345    | 

##### 4. Print the final testing PSNR at convergence [20pt]

- `*** NOTE ***` The values should be presented up to 5 decimal places (소수점 5째 자리까지 표기하시오)

| PSNR              |            |
| ----------------- | ---------- |
| training          | 20.00000   | 
| testing           | 20.00000   | 

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

##### 5. Print the testing PSNR within the last 10 epochs [5pt]

- `*** NOTE ***` The values should be presented up to 5 decimal places (소수점 5째 자리까지 표기하시오)

[epoch = 0991] 19.10000  
[epoch = 0992] 19.20000  
[epoch = 0993] 19.30000  
[epoch = 0994] 19.40000  
[epoch = 0995] 19.50000  
[epoch = 0996] 19.60000  
[epoch = 0997] 19.70000  
[epoch = 0998] 19.80000  
[epoch = 0999] 19.90000  
[epoch = 1000] 20.00000