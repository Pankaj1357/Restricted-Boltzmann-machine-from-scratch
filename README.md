# Restricted-Boltzmann-machine-from-scratch
Followings things are implemented in this repository:-

<p><b>=></b> Implemented Gibbs sampling in Restrictive Boltzmann Machine (RBM) and trained it using
weights using contrastive divergence. MNIST was used as the dataset and the model was implemented using only numpy library.</p>
<p><b>=></b> Input size, hidden layer size, and number of steps (k) in the Gibbs chain to be variables, so that these can be easily changed.</p>
<p><b>=></b> L2 weight decay and momentum in the implementation of training using contrastive divergence.</p>
<p><b>=></b> Visualized the evolution of MNIST digits in RBM after some k steps. That is, after training, produced some figures such that first figure in the row is a test image, andthen the next three images in the row are fantasy images for for k = 3 starting with the given test image.</p>
<p><b>=></b> Compare direct classfication of pixels and learned RBM features by training a Softmax classifier with L2 weight decay</b>
