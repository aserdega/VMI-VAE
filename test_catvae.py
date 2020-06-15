#from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import os
import sys
import shutil
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import tensorboardX

from trainer import *
from utils import get_config, get_mnist_data_loader, get_fashion_mnist_data_loader, prepare_sub_folder, Timer, write_images, write_loss

OUT_DIR = './cvae_tests'

def plot_hists_for_one_hots(onehot_counts):
    X = np.array(range(10))
    n_onehot = onehot_counts.size(0)
    for i in range(n_onehot):
        counts = onehot_counts[i,:].cpu().numpy()
        plt.bar(X,counts, align='center')
        plt.title('One hot {}'.format(i + 1))
        plt.xlabel('Image digit') 
        plt.savefig(OUT_DIR + '/onehot_hist_{}.png'.format(i))
        plt.clf()

def manipulate_continious(idx, dim, recons, samples, file_name):
    new_sample = samples.clone().detach()
    sample_list = [recons[idx:idx+1]]
    for i in range(11):
        new_sample.data[idx:idx+1,dim] = -5. + i
        out = trainer.vae.decoder(new_sample)
        sample_list.append(out[idx:idx+1])

    concat_manipul_recons = torch.cat(sample_list,0)
    vutils.save_image(concat_manipul_recons/2. + 0.5, file_name, nrow=12)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnistcat.yaml', help='Path to the config file.')
parser.add_argument('--experiment_id', type=str, help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--dataset', type=str, default='mnist', help="dataset")
parser.add_argument("--no_accuracy", action="store_true")
parser.add_argument("--no_discrete", action="store_true")
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
model_name           = ('mnistcat' if opts.dataset == 'mnist' else 'fsmnistcat') + opts.experiment_id
output_directory     = os.path.join(opts.output_path + "/outputs_cat", model_name)
checkpoint_directory = os.path.join(output_directory, 'checkpoints')

saved_config = os.path.join(output_directory,'config.yaml')
config = get_config(saved_config)

max_iter     = config['max_iter']
display_size = config['display_size']

imgconf = config['image']
disconf = config['dis']

# Setup model and data loader
config['num_workers'] = 2
if ('data' not in config) or config['data'] == 'mnist':
    train_loader, test_loader = get_mnist_data_loader(config)
elif config['data'] == 'fsmnist':
    train_loader, test_loader = get_fashion_mnist_data_loader(config)

if config['mode'] == 'CatVAE':
    print('Testing CatVAE')
    trainer = TrainerCatVAE(config)
elif config['mode'] == 'InfoCatVAE':
    print('Testing InfoCatVAE')
    trainer = TrainerInfoCatVAE(config)
else:
    sys.exit("Only support CatVAE or  InfoCatVAE")
trainer.cuda()


iterations = 0
print('Loading model from {}'.format(checkpoint_directory))
trainer.resume(checkpoint_directory, hyperparameters=config)

prior_samples = trainer.vae.sample_full_prior(config['batch_size'])

trainer.vae.train()

with torch.no_grad():
    decoded_samples = trainer.vae.decoder(prior_samples)
    #save generated samples
    file_name = OUT_DIR + '/cvae_generated.jpg'
    vutils.save_image(decoded_samples[:100]/2 + 0.5, file_name, nrow=10)

    idx = 3
    #manipulate codes of generated samples
    #manipulate first continious code
    manipulate_continious(idx,0,decoded_samples,prior_samples,OUT_DIR+'/cvae_0_cont_code_manipul_gen.jpg')
    #manipulate second continious code
    manipulate_continious(idx,1,decoded_samples,prior_samples,OUT_DIR+'/cvae_1_cont_code_manipul_gen.jpg')


    dataset_list = list(train_loader)
    images = dataset_list[0][0].cuda()
    labels = dataset_list[0][1].cuda()

    recons, enc_samples = trainer.vae.encode_decode(images,hard_catg=True)
    enc_samples_backup = enc_samples.clone().detach()

    #just save reconstructions and originals
    write_images([recons,images],
                  11, OUT_DIR,
                  '_input_recons')

    idx = 1
    #manipulate codes of encoded samples
    #manipulate first continious code
    manipulate_continious(idx,0,recons,enc_samples,OUT_DIR+'/cvae_0_cont_code_manipul_rec.jpg')
    #manipulate second continious code
    manipulate_continious(idx,1,recons,enc_samples,OUT_DIR+'/cvae_1_cont_code_manipul_rec.jpg')

    if opts.no_discrete:
        sys.exit('Done')

    #manipulate discrete code of encoded images and then decode
    out_list = []
    for s in range(40):
        out_list.append(recons[s:s+1])
        enc_samples = enc_samples_backup.clone().detach()
        for i in range(10):
            for j in range(10):
                enc_samples[s,-(1+j)] = 1. if j == i else 0.
            recons = trainer.vae.decoder(enc_samples)
            out_list.append(recons[s:s+1])

    all_samples = torch.cat(out_list,0)

    file_name = OUT_DIR + '/cvae_recon_manipul.jpg'
    vutils.save_image(all_samples/2 + 0.5, file_name, nrow=11)

    #manipulate discrete code of prior samples and and then decode
    out_list = []
    for s in range(10):
        out_list.append(decoded_samples[s:s+1])
        for i in range(10):
            for j in range(10):
                prior_samples[s,-(1+j)] = 1. if j == i else 0.
            geners = trainer.vae.decoder(prior_samples)
            out_list.append(geners[s:s+1])

    all_samples = torch.cat(out_list,0)

    file_name =  OUT_DIR + '/cvae_sampl_manipul.jpg'
    vutils.save_image(all_samples/2 + 0.5, file_name, nrow=11)

    if opts.no_accuracy:
        sys.exit('Done')

    counter = torch.zeros(10,10).cuda()
    average = torch.zeros(10).cuda()
    count   = torch.zeros(10).cuda()
    batch_size = config['batch_size']

    for d in dataset_list:
        images = d[0].cuda().detach()
        labels = d[1].cuda().detach()

        recons, enc_samples = trainer.vae.encode_decode(images,hard_catg=True)

        embeddings = enc_samples[:,-10:]
        embeddings_arg_max = torch.argmax(embeddings,1)

        for s in range(batch_size):
            average[embeddings_arg_max[s]] += labels[s]
            count[embeddings_arg_max[s]]   += 1
            counter[embeddings_arg_max[s],labels[s]] += 1 

    lookup = torch.argmax(counter,1)
    plot_hists_for_one_hots(counter)
    print('Which digit were mostly ecoded into the onehot')
    print(lookup)

    count = 0
    correct = 0

    for d in dataset_list:
        images = d[0].cuda().detach()
        labels = d[1].cuda().detach()

        recons, enc_samples = trainer.vae.encode_decode(images,hard_catg=True)

        embeddings = enc_samples[:,-10:]
        embeddings_arg_max = torch.argmax(embeddings,1)

        for s in range(batch_size):
            count += 1
            correct += 1. if lookup[embeddings_arg_max[s]] == labels[s] else 0.

    print('Classif accuracy: {}'.format(correct / count))

sys.exit('Done')
