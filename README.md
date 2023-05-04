Download Link: https://assignmentchef.com/product/solved-cs434-assignment-1-linear-regression
<br>



<strong>Data </strong>You will use the Boston Housing dataset of the housing prices in Boston suburbs. The goal is to predict the median value of housing of an area (in thousands) based on 13 attributes describing the area (e.g., crime rate, accessibility etc). The file housing desc.txt describes the data. Data is divided into two sets: (1) a training set housing train.csv for learning the model, and (2) a testing set housing test.csv for evaluating the performance of the learned model. Your task is to implement linear regression and explore some variations with it on this data.

<ol>

 <li>(10 pts) Load the training data into the corresponding <em>X </em>and <em>Y </em>matrices, where <em>X </em>stores the features and <em>Y </em>stores the desired outputs. The rows of <em>X </em>and <em>Y </em>correspond to the examples and the columns of <em>X </em>correspond to the features. Introduce the dummy variable to <em>X </em>by adding an extra column of ones to <em>X </em>(You can make this extra column to be the first column. Changing the position of the added column will only change the order of the learned weight and does not matter in practice). Compute the optimal weight vector <strong>w </strong>using <strong>w </strong>= (<em>X<sup>T</sup>X</em>)<sup>−1</sup><em>X<sup>T</sup>Y </em>. Feel free to use existing numerical packages (e.g., numpy) to perform the computation. Report the learned weight vector.</li>

 <li>(10 pts) Apply the learned model to make predictions for the training and testing data respectively and compute for each case the average squared error(ASE), defined by 1, which is the sum of squared error normalized by <em>n</em>, the total number of examples in the data. Report the training and testing ASEs respectively. Which one is larger? Is it consistent with your expectation?</li>

</ol>

Write your code so that you get the results for questions 1 and 2 using the following command: <em>python q1 2.py housing train.csv housing test.csv </em>The output should include:

<ul>

 <li>the learned weight vector</li>

 <li>ASE over the training data</li>

 <li>ASE over the testing data</li>

</ul>

<ol start="3">

 <li>(10 pts) Remove the dummy variable (the column of ones) from <em>X</em>, repeat 1 and 2. How does this change influence the ASE on the training and testing data? Provide an explanation for this influence.</li>

</ol>

Write your code so that you get the results for question 3 using the following command: <em>python q1 3.py housing train.csv housing test.csv </em>The output should include:

<ul>

 <li>the learned weight vector</li>

 <li>ASE over the training data</li>

 <li>ASE over the testing data</li>

</ul>

<ol start="4">

 <li>(20 pts) Modify the data by adding additional random features. You will do this to both training and testing data. In particular, generate 20 random features by sampling from a standard normal distribution. Incrementally add the generated random features to your data, 2 at a time. So we will create 20 new train/test datasets, each with <em>d </em>of random features, where <em>d </em>= 2<em>,</em>4<em>,…,</em> For each version, learn the optimal linear regression model (i.e., the optimal weight vector) and compute its resulting training and testing ASEs. Plot the training and testing ASEs as a function of <em>d</em>. What trends do you observe for training and testing ASEs respectively? In general, how do you expect adding more features to influence the training ASE? How about testing ASE? Why?</li>

</ol>

Write your code so that you get the results for question 4 using the following command: <em>python q1 4.py housing train.csv housing test.csv </em>The output should include:

<ul>

 <li>plot of the training ASE (y-axis) as a function of d (x-axis)</li>

 <li>plot of the testing ASE (y-axis) as a function of d (x-axis)</li>

</ul>

<h1>2           Logistic regression with regularization (to come)</h1>

<strong>Data. </strong>For this part you will work with the USPS handwritten digit dataset and implement the logistic regression classifier to differentiate digit 4 from digit 9. Each example is an image of digit 4 or 9, with 16 by 16 pixels. Treating the gray-scale value of each pixel as a feature (between 0 and 255), each example has 16<sup>2 </sup>= 256 features. For each class, we have 700 training samples and 400 testing samples. For this assignment, we have injected some small amount of salt and pepper noise to the image. You can view the original images collectively at <a href="http://www.cs.nyu.edu/~roweis/data/usps_4.jpg">http://www.cs.nyu.edu/</a><a href="http://www.cs.nyu.edu/~roweis/data/usps_4.jpg">~</a><a href="http://www.cs.nyu.edu/~roweis/data/usps_4.jpg">roweis/data/usps_4.jpg</a><a href="http://www.cs.nyu.edu/~roweis/data/usps_4.jpg">,</a> and<a href="http://www.cs.nyu.edu/~roweis/data/usps_9.jpg">http://www.cs. </a><a href="http://www.cs.nyu.edu/~roweis/data/usps_9.jpg">nyu.edu/</a><a href="http://www.cs.nyu.edu/~roweis/data/usps_9.jpg">~</a><a href="http://www.cs.nyu.edu/~roweis/data/usps_9.jpg">roweis/data/usps_9.jpg</a> The data is in the csv format and each row corresponds to a handwritten digit (the first 256 columns) and its label (last column, 0 for digit 4 and 1 for digit 9).

<ol>

 <li>(20 pts) Implement the batch gradient descent algorithm to train a binary logistic regression classifier. The behavior of Gradient descent can be strongly influenced by the learning rate. Experiment with different learning rates, report your observation on the convergence behavior of the gradient descent algorithm. For your implementation, you will need to decide a stopping condition. You might use a fixed number of iterations, the change of the objective value (when it ceases to be significant) or the norm of the gradient (when it is smaller than a small threshold). Note, if you observe an overflow, then your learning rate is too big, so you need to try smaller (e.g., divide by 2 or 10) learning rates. Once you identify a suitable learning rate, rerun the training of the model from the beginning. For each gradient descent iteration, plot the training accuracy and the testing accuracy of your model as a function of the number of gradient descent iterations. What trend do you observe? Write your code so that you get the results for question 1 using the following command: <em>python q2 1.py usps train.csv usps test.csv learningrate </em>The output should include:

  <ul>

   <li>plot of the learning curve: training accuracy (y-axis) as a function of the number of gradient descent iterations (x-axis)</li>

   <li>plot of the learning curve: testing accuracy (y-axis) as a function of the number of gradient descent iterations (x-axis)</li>

  </ul></li>

 <li>(10 pts) Logistic regression is typically used with regularization. Here we will explore <em>L</em><sub>2 </sub>regularization, which adds to the logistic regression objective an additional regularization term of the squared Euclidean norm of the weight vector.</li>

</ol>

where the loss function <em>l </em>is the same as introduced in class. Find the gradient for this objective function and modify the batch gradient descent algorithm with this new gradient. Provide the pseudo code for your modified algorithm.

<ol start="3">

 <li>(25 pts) Implement your derived algorithm, and experiment with different <em>λ </em>values (e.g., 10</li>

</ol>

Report the training and testing accuracies (i.e., the percentage of correct predictions) achieved by the weight vectors learned with different <em>λ </em>values. Discuss your results in terms of the relationship between training/testing performance and the <em>λ </em>values. Write your code so that you get the results for question 3 using the following command: <em>python q2 </em><em>3.py usps train.csv usps </em><em>test.csv lambdas</em>

where <em>lambdas </em>contains the list of <em>λ </em>values to be tested. The output should include:

<ul>

 <li>plot of the training accuracy (y-axis) as a function of the <em>λ </em>value (x-axis)</li>

 <li>plot of the testing accuracy (y-axis) as a function of the <em>λ </em>value (x-axis)</li>

</ul>

<strong>Remark 1 </strong><em>For logistic regression, it would be a good idea to normalize the features to the range of </em>[0<em>,</em>1]<em>. This will makes it easier to find a proper learning rate. You can find information about feature normalization at </em><em><a href="https://en.wikipedia.org/wiki/Feature_scaling">https://en.wikipedia.org/wiki/Feature_scaling</a><a href="https://en.wikipedia.org/wiki/Feature_scaling">)</a></em>