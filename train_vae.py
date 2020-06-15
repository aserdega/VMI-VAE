#from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import os
import sys
import shutil
import argparse

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import tensorboardX

from trainer import *
from utils import get_config, get_mnist_data_loader, prepare_sub_folder, Timer, write_images, write_loss

from networks import QNet


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnist.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='VAE', help="VAE|VAEGAN|InfoVAEGAN|InfoVAE")
parser.add_argument('--seed', type=int, default=1, help='Seed')
opts = parser.parse_args()


if opts.seed != -1:
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    print('Seed: {0}'.format(opts.seed))
cudnn.benchmark = True


# Load experiment setting
config = get_config(opts.config)


print('kl_w :{}'.format(config['kl_w']))
print('adv_w:{}'.format(config['adv_w']))
print('inf_w:{}'.format(config['inf_w']))

max_iter     = config['max_iter']
display_size = config['display_size']


imgconf = config['image']
disconf = config['dis']


# model and data loader
train_loader, test_loader = get_mnist_data_loader(config)

if config['mode'] == 'VAEGAN':
    print('Training VAEGAN')
    trainer = TrainerVAEGAN(config)
elif config['mode'] == 'InfoVAEGAN':
    print('Training InfoVAEGAN')
    trainer = TrainerInfoVAEGAN(config)
elif config['mode'] == 'InfoVAE':
    print('Training InfoVAE')
    trainer = TrainerInfoVAE(config)
elif config['mode'] == 'VAE':
    print('Training VAE')
    trainer = TrainerVAE(config)
else:
    sys.exit("Only support VAE, VAEGAN or InfoVAEGAN")
trainer.cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0] + config['experiment_id']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Start training
iterations = 0
prior_samples = trainer.vae.prior.sample_prior(config['batch_size'])

while True:
    for it, (images, _) in enumerate(train_loader):
        trainer.update_learning_rate()

        #training happens here
        images = images.cuda().detach()
        reconstructed = trainer.update_vae(images, config)
        if trainer.update_dis is not None:
            trainer.update_dis(images, config)
        torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        #save some image stuff
        if (iterations + 1) % config['image_save_iter'] == 0 or iterations == 0:
            with torch.no_grad():
                generated_images = trainer.vae.decoder(prior_samples)

                write_images([images,reconstructed,generated_images],
                              display_size,image_directory,
                              'train_%08d'%(iterations + 1))
                #visualize latent code influence on output
                if (iterations + 1) % (config['image_save_iter'] * 2) == 0:
                    trainer.get_latent_visualization(image_directory,'train_%08d'%(iterations + 1),images,prior_samples)

        iterations += 1
        if iterations >= max_iter:
            if trainer.save is not None:
                trainer.save(checkpoint_directory, iterations)
            sys.exit('Finished training')
