# Restricted-Boltzmann-machine-from-scratch
Followings things are implemented in this repository:-

<p><b>=></b> Implemented Gibbs sampling in Restrictive Boltzmann Machine (RBM) and trained it using
weights using contrastive divergence. MNIST was used as the dataset and the model was implemented using only numpy library.</p>
<p><b>=></b> Input size, hidden layer size, and number of steps (k) in the Gibbs chain to be variables, so that these can be easily changed.</p>
<p><b>=></b> L2 weight decay and momentum in the implementation of training using contrastive divergence.</p>
<p><b>=></b> Visualized the evolution of MNIST digits in RBM after some k steps. That is, after training, produced some figures such that first figure in the row is a test image, andthen the next three images in the row are fantasy images for for k = 3 starting with the given test image.</p>
<p><b>=></b> Compare direct classfication of pixels and learned RBM features by training a Softmax classifier using Cross Entropy Loss with L2 weight decay</b>

# Running the code

<b>1# Getting the Dataset :::</b>


<p>Download the zip file from  ::::    http://yann.lecun.com/exdb/mnist/ <br>
Download the file names  :::  train-images-idx3-ubyte.gz     |and|       train-labels-idx1-ubyte.gz <br>
unzip the file in current directory and run this script(mnist.py) to save the .csv file in current directory</p>




<p><b>2#</b> Once dataset is obtained in .csv file make sure that your current directory have following files 

   --- main.ipynb<br>
   --- mnist_train.csv<br>
   --- softmax_classifier.py<br>
   --- utilities.py
 </p>   
   
<p><b>3#</b> Now run 'main.ipynb' file can be run cell by cell to replicate the results.</p>
   
   
  

