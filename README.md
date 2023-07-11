# Knowledge-Distillation-for-Discriminator-of-SRGAN

A TensorFlow implementation of the paper "A Study of Lightening SRGAN Using Knowledge Distillation".  
This code is built on [super-resolution](https://github.com/krasserm/super-resolution.git). Thanks to the author for sharing the codes.

### Abstract
Recently, convolutional neural networks (CNNs) have been widely used with excellent performance in various computer vision fields, including super-resolution (SR). However, CNN is computationally intensive and requires a lot of memory, making it difficult to apply to limited hardware resources such as mobile or Internet of Things devices. To solve these limitations, network lightening studies have been actively conducted to reduce the depth or size of pre-trained deep CNN models while maintaining their performance as much as possible. This paper aims to lighten the SR CNN model, SRGAN, using the knowledge distillation among network lightening technologies; thus, it proposes four techniques with different methods of transferring the knowledge of the teacher network to the student network and presents experiments to compare and analyze the performance of each technique. In our experimental results, it was confirmed through quantitative and qualitative evaluation indicators that student networks with knowledge transfer performed better than those without knowledge transfer, and among the four
knowledge transfer techniques, the technique of conducting adversarial learning after transferring knowledge from the teacher generator to the student generator showed the best performance

### Dataset
I use DIV2K dataset as a training set and use four benchmark datasets (Set5, Set14, BSD100, Urban100).  
You can download these datasets from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [benchmarks](https://cv.snu.ac.kr/research/EDSR/benchmark.tar).

### Pre-trained weights
Make a directory called 'weights' and place the weights in it.  
You can download SRGAN weights from [weights](https://martin-krasser.de/sisr/weights-srgan.tar.gz) or train the SRGAN model to get the weights.  

### Usage
Train a teacher model and save the weights
```
python teacher.py
```
Train a student model to get a generator model,
```
python student.py
```
and train each student to distill the knowledge.
```
python <GAN2G, G2G+AL, ...>.py
```

### Note

