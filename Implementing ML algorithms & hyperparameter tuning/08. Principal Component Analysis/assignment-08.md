# Principal Component Analysis

#### 1. Data

- the data are given by the file `data-pca.txt`
- the data consist of a set of points $`\{ (x_i, y_i) \}_{i=1}^{n}`$ where $`z_i = (x_i, y_i)`$ denotes a 2-dimensional point in the cartesian coordinate

#### 2. Normalization

- the data is normalized to have the mean = 0 and the standard deviation = 1

    $`x = \frac{x - \mu_x}{\sigma_x}`$ and $`y = \frac{y - \mu_y}{\sigma_y}`$
    - $`\mu_x`$ denotes the mean of $`x`$
    - $`\sigma_x`$ denotes the standard deviation of $`x`$
    - $`\mu_y`$ denotes the mean of $`y`$
    - $`\sigma_y`$ denotes the standard deviation of $`y`$

#### 3. Covariance Matrix

- compute the co-variance matrix 

    $`\Sigma = \frac{1}{n} \sum_{i = 1}^n z_i z_i^T = \frac{1}{n} Z^T Z`$
    - $`n`$ denotes the number of data
    - $`Z = \begin{bmatrix} z_1^T \\ \vdots \\ z_n^T \end{bmatrix}`$

#### 4. Principal Components

- compute the eigen-values and the eigen-vectors of the co-variance matrix

## Code

- load the data from the file

``` python
data = np.loadtxt('data-pca.txt', delimiter=',')

x = data[:,0]
y = data[:,1]
```

- define a function to normalize the input data points $`x`$ and $`y`$

``` python
def normalize_data(x, y):

    xn = # normalize x. the mean of xn is zero and the standard deviation of xn is one #
    yn = # normalize y. the mean of yn is zero and the standard deviation of yn is one #

    return xn, yn
```

- define a function to compute the co-variance matrix of the data

``` python
def compute_covariance(x, y):

    covar = # compute the covariance matrix #
    
    return covar
```

- define a function to compute the principal directions from the co-variance matrix

``` python
def compute_principal_direction(covariance):

    direction = # compute the principal directions from the co-variance matrix #
    
    return direction
```

- define a function to compute the projection of the data point onto the principal axis

``` python
def compute_projection(point, axis):

    projection = # compute the projection of point on the axis #
    
    return projection
```

- define a function to compute the projection of the data point onto the principal axis

``` python
def compute_distance(point1, point2):

    distance = # compute the Euclidean distance between point1 and point2 #
    
    return distance
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

##### 1. Plot the original data points [1pt]

<img src="data-original.png"  width="600">

##### 2. Plot the normalized data points [1pt]
- $`z = \frac{z - \mu}{\sigma}`$
- $`\mu`$ denotes the average and $`\sigma`$ denotes the standard deviation

<img src="data-normalize.png"  width="600">

##### 3. Plot the principal axes [2pt]
- plot the normalized data points 
- plot the first principal vector 
- plot the second principal vector

<img src="principal-direction.png"  width="600">

##### 4. Plot the first principal axis [3pt]
- plot the normalized data points 
- plot the first principal axis 

<img src="principle-axis-1.png"  width="600">

##### 5. Plot the project of the normalized data points onto the first principal axis [4pt]
- plot the normalized data points 
- plot the first principal axis 
- plot the projected points from the normalized data points onto the first principal axis

<img src="principle-axis-1-projection.png"  width="600">

##### 6. Plot the lines between the normalized data points and their projection points on the first principal axis [3pt]
- plot the normalized data points 
- plot the first principal axis 
- plot the projected points from the normalized data points onto the first principal axis
- plot the lines that connect between the normalized data points and their projection points on the first principal axis

<img src="principle-axis-1-distance.png"  width="600">

##### 7. Plot the second principal axis [3pt]
- plot the normalized data points 
- plot the second principal axis 

<img src="principle-axis-2.png"  width="600">

##### 8. Plot the project of the normalized data points onto the second principal axis [4pt]
- plot the normalized data points 
- plot the second principal axis 
- plot the projected points from the normalized data points onto the second principal axis

<img src="principle-axis-2-projection.png"  width="600">

##### 9. Plot the lines between the normalized data points and their projection points on the second principal axis [3pt]
- plot the normalized data points 
- plot the second principal axis 
- plot the projected points from the normalized data points onto the second principal axis
- plot the lines that connect between the normalized data points and their projection points on the second principal axis

<img src="principle-axis-2-distance.png"  width="600">