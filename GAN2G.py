#Distill Teacher to Student only Gen and teacher knowledge = Gen + Dis 
import os
import matplotlib
import matplotlib.pyplot as plt
from data import DIV2K
from model.srgan import generator, discriminator, st_generator
from train import Train_and_Distill_only_gen_noD
from model import resolve_single
from utils import load_image, plot_sample
from teacher import teacher_gan_generator
from model import evaluate
import PIL.Image
import numpy as np
import tensorflow
matplotlib.pyplot.ion()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)
os.makedirs(weights_dir, exist_ok=True)

#dataset
div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')
train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

#generator fine-tuning
gan_generator = st_generator()
gan_generator.load_weights(weights_file('st_pre_generator.h5'))
teacher_gan_generator.load_weights(weights_file('teacher_gan_generator.h5'))

#train_only_gen
gan_trainer = Train_and_Distill_only_gen_noD(generator=gan_generator, teacher=teacher_gan_generator)
gan_trainer.train(train_ds, steps=1000)
gan_trainer.generator.save_weights(weights_file('gen3_gan_generator.h5'))

#Demo
gan_generator = st_generator()
gan_generator.load_weights(weights_file('gen3_gan_generator.h5'))

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
    
    
'''my_path = os.path.abspath('SAVE_PATH')
def resolve_and_plot(lr_image_path, path):
    rgba_image=PIL.Image.open(lr_image_path)
    rgb_image = rgba_image.convert('RGB')
    lr=rgb_image
    
    gan_sr = resolve_single(gan_generator, lr)
    a = tensor_to_image(gan_sr)
    
    plt.figure(figsize=(11.05,11.05),frameon=False)
    images = [gan_sr]
    positions = [1]
    my_file =f'{path}.png'
    for i, (img, pos) in enumerate(zip(images, positions)):
        plt.axis('off')
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(my_path, my_file), bbox_inches='tight',pad_inches = 0)
    
resolve_and_plot(f'TEST_PATH', DATA_NAME)'''