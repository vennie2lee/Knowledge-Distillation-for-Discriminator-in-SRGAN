#original teacher and original student train
import os
import matplotlib
import matplotlib.pyplot as plt
from data import DIV2K
from model.srgan import generator, discriminator, st_generator
from train import SrganTrainer, SrganGeneratorTrainer
from model import resolve_single, evaluate
from utils import load_image, plot_sample
import numpy as np
import math
import tensorflow
from tensorflow import keras
import PIL.Image
import cv2
matplotlib.pyplot.ion()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)
os.makedirs(weights_dir, exist_ok=True)

# dataset
div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

# Generator fine-tuning(GAN)
teacher_gan_generator = generator()
teacher_gan_generator.load_weights(weights_file('pre_generator.h5'))

gan_trainer = SrganTrainer(generator=teacher_gan_generator, discriminator=discriminator())
gan_trainer.train(train_ds, steps=1000)
gan_trainer.generator.save_weights(weights_file('teacher_gan_generator.h5'))
gan_trainer.discriminator.save_weights(weights_file('teacher_gan_discriminator.h5'))

#Demo
teacher_gan_generator = generator()
teacher_gan_generator.load_weights(weights_file('teacher_gan_generator.h5'))