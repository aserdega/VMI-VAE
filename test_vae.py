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


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnist.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='VAE', help="VAE|VAEGAN|InfoVAEGAN")
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

max_iter     = config['max_iter']
display_size = config['display_size']

imgconf = config['image']
disconf = config['dis']

# Setup model and data loader
train_loader, test_loader = get_mnist_data_loader(config)

if config['mode'] == 'VAEGAN':
    print('Training VAEGAN')
    trainer = TrainerVAEGAN(config)
elif config['mode'] == 'InfoVAEGAN':
    print('Training InfoVAEGAN')
    trainer = TrainerInfoVAEGAN(config)
elif config['mode'] == 'InfoVAE':
    print('Testing InfoVAE')
    trainer = TrainerInfoVAE(config)
elif config['mode'] == 'VAE':
    print('Testing VAE')
    trainer = TrainerVAE(config)
else:
    sys.exit("Only support VAE, VAEGAN or InfoVAEGAN")

trainer.cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0] + config['experiment_id']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Start training
iterations = 0
print('Loading model from {}'.format(checkpoint_directory))
trainer.resume(checkpoint_directory, hyperparameters=config)
prior_samples = trainer.vae.prior.sample_prior(config['batch_size'])

trainer.vae.train()

new_sample = prior_samples.clone().detach()
sample_list = []
with torch.no_grad():
    for i in range(11):
        new_sample.data[0:1,0] = -5. + i
        out = trainer.vae.decoder(new_sample)
        sample_list.append(out[0:1])

all_samples = torch.cat(sample_list,0)
write_images([all_samples],
              11,'.',
              '_test')

################################
#generate grid from latent space
################################

idx = 11
rng = 10
start = 3
step = start*2./rng

sample_list = []
target_sample = prior_samples.clone().detach()

with torch.no_grad():
    for i in range(rng):
        target_sample = prior_samples.clone().detach()
        target_sample.data[idx,0] = -start + i * step
        for j in range(rng):
            target_sample.data[idx,1] = -start + j * step
            out = trainer.vae.decoder(target_sample)
            sample_list.append(out[idx:idx+1])


all_samples = torch.cat(sample_list,0)

file_name = '%s/vae_recgen%s.jpg' % ('.', '_test_grid')
vutils.save_image(all_samples/2 + 0.5, file_name, nrow=rng)

sys.exit('Done!')
