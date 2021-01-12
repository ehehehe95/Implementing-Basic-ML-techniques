# K-means clustering

#### 1. Data

- the data are given by the file `data-kmeans.csv`
- the data consist of a set of points $`\{ (x_i, y_i) \}_{i=1}^{n}`$ where $`z_i = (x_i, y_i)`$ denotes a 2-dimensional point in the cartesian coordinate and $`n`$ is given as $`200`$

#### 2. Loss

- the loss function $`\mathcal{L}(C_1, C_2, \cdots, C_k, \mu_1, \mu_2, \cdots, \mu_k)`$ with a given number of clusters $`k`$ for a set of data $`\{ z_i \}_{i=1}^{n}`$ is defined by:

    $` \mathcal{L}(C_1, C_2, \cdots, C_k, \mu_1, \mu_2, \cdots, \mu_k) = \frac{1}{n} \sum_{i=1}^n \| z_i - \mu_{l(z_i)} \|_2^2 = \frac{1}{n} \sum_{j=1}^k \sum_{z_i \in C_j} \| z_i - \mu_j \|_2^2 `$

    - $`l(z) = k`$ is a label function that defines a label $`k`$ of point $`z`$
    - $`C_k`$ denotes a set of points $`\{ z_i | l(z_i) = k \}`$ of label $`k`$
    - $`\mu_k`$ denotes a centroid of points in $`C_k`$

#### 3. Optimisation

- the label $`l(z)`$ of each point $`z`$ is determined by:

    $`l(z) = \arg\min_k \| z - \mu_k \|_2^2`$

- the centroid $`\mu_i`$ of cluster $`k`$ is determined by:

    $`\mu_k = \frac{\sum_{z_i \in C_k} z_i}{|C_k|}`$

#### 4. clustering

- initialise labels $`l(z_i)`$ for point $`z_i`$ for all $`i`$ randomly
- optimise the loss function with respect to the centroids and the clusters in an alternative way
- set the number of clusters $`k = 5`$


## Code

- load the data from the files

``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data-kmeans.csv')
data = dataset.values
```

- define a function to compute a distance between two points $`a`$ and $`b`$

``` python
def compute_distance(a, b):

    dist = #distance between a and b#

    return dist
```

- define a function to compute a centroid from a given set of points $`Z`$

``` python
def compute_centroid(Z):

    center = #centroid of a set of points in P#
    
    return center
```

- define a function to determine the label of point $`z`$ with a set of centroids $`M`$

``` python
def compute_label(z, M):

    label = #label of point z with a set of centroids M#
    
    return label
```

- define a function to compute the loss with a set of clusters $`C`$ and a set of centroids $`M`$

``` python
def compute_loss(C, M):

    loss = #compute loss#
    
    return loss
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

##### 1. Plot the data points [1pt]

<img src="kmeans-data.png"  width="600">

##### 2. Visualise the initial condition of the point labels [1pt]
- initialise the label of each point randomly
- visualise the centroid of each cluster as well
- use different colors for different cluster labels

<img src="kmeans-initial.png"  width="600">

##### 3. Plot the loss curve [5pt]
- x-axis represents optimisation iteration
- y-axis represents the loss value

<img src="kmeans-loss.png"  width="600">

##### 4. Plot the centroid of each clsuter [5pt]
- x-axis represents the optimisation iteration
- y-axis represents the distance from the origin to each centroid
- use idfferent colors for different cluster labels

<img src="kmeans-centroid.png"  width="600">

##### 5. Plot the final clustering result [5pt]
- plot the data points with their obtained cluster labels
- visualise the centroid of each cluster as well
- use different colors for different cluster labels

<img src="kmeans-final.png"  width="600">