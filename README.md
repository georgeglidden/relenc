# The Relational Encoder Architecture

## Overview
Introduced in [1], the _Relational Encoder_ is a machine learning model and training paradigm which cultivates a meaningful feature-space representation of a dataset by optimizing a bipartite architecture. The _encoder_ network can be any trainable model associating inputs with output vectors in a feature space, e.g., several convolutional layers appended with a multilayer perceptron. Given an encoder with well-defined output dimensions, the _relation head_ is a secondary MLP which assigns a _similarity score_, between 0 and 1, to any pair of feature vectors.

The significance of the relational encoder training scheme lies in its _unsupervised_ implementation. The encoder learns a generalized version of the data, and the resulting feature space reflects the implicit structure of the dataset. No external biases need be introduced, nor are any significant preprocessing steps required. Furthermore, the relation head serves as an informed comparison operator, regardless of dimensionality or quality of the input, and is capable of enhancing a variety of algorithms which consider pairwise similarities in data.

## Training Cycle
The `encoder` and `relator` models are trained simultaneously on batched data. Given a batch size `m` and features vectors with `d` values, the batch is encoded to produce a tensor of dimension `m*d`. For each vector in the encoded tensor, `2k*k` pairs are created and autolabeled. The first `k*k` pairs are sampled from a sequence of randomized transformations applied to the vector itself, and for this reason are labeled as `positive` pairs with target similarity `1`. Conversely, the next `k*k` pairs are sampled from the other `m-1` feature vectors, and labeled as `negative` with target similarity `0`.

The `relator` assigns similarity scores to each pair, and the binary crossentropy between these values and the target values is backpropagated through both networks.

Clearly, the autolabeling scheme is not perfect. Given sufficiently many batches or a sufficiently small number of true classes in the data, false negatives will be generated. However, the distribution  of these errors is uniform, their density is low, and so their effect on training is negligible\*, and empirical results show that convergence does occur.

\* Every aggregate batch represents `m` classes. Every class has multiple positive pairs in the aggregation. So, even if false negatives for that class appear, the effect on the batch gradient will be analogous to reducing the batch size by the number of false negatives. No chaos is introduced to the training dynamics.

## Example
Relational Encoder examples, implementated in Tensorflow and Pytorch, are provided in the `/examples` subdirectory.

# Primary Goal:
## Classifying pre-extracted frames from CryoEM data (semisupervised)
In this project, the relational encoder architecture is applied to particles extracted from micrographs. Our intent is to produce a software tool which, given a set of numeric arrays - in this case, images - generates a natural classification on that data. The input set is ultimately partitioned into `k` subsets, each of which contain elements of a common class.

This method is partially supervised, as we do not provide a means for extracting particle-containing frames.

A python application satisfying this goal, as well as documentation and a user mannual, will be made available in the `/app` subdirectory. NOTE: this project is a work in progress, and - to reiterate a passage of the license - is provided as-is!

## Secondary Goals (unsupervised)
The relational encoder can [should be able to] be used for tasks more difficult and complex than classification. Several interactive demonstrations of this potential will be made available in the `/notebooks` subdirectory.

# Attribution
This project was made possible by, and produced on behalf of, RTB-NIAID and Axle Informatics.

## Contact Info
george.glidden@axleinfo.com

gliddenegeorge@gmail.com

george.glidden@umconnect.umt.edu

https://github.com/georgeglidden

# Citations
[1] Patacchiola, M. and Storkey, A., 2021. Self-Supervised Relational Reasoning for Representation Learning. NeurIPS Proceedings. https://papers.nips.cc/paper/2020/hash/29539ed932d32f1c56324cded92c07c2-Abstract.html

https://caffe.berkeleyvision.org/

https://onnx.ai/get-started.html
