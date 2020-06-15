import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from networks import VAE, DiscriminatorVAE, DiscriminatorInfoVAE, QNet, CatVAE, CatQNet
from utils import weights_init, get_scheduler, get_model_list

import torchvision.utils as vutils

import numpy as np

#trains vanilla vae
class TrainerVAE():
    def __init__(self, config):
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = VAE(imgconf,config['gen'],config['latent'])
        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)
        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        self.mse_crit = nn.MSELoss()

    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,config):
        #passes adv grad to decoder
        self.vae_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        # vae part update
        recons, latent, samples = self.vae(inputs)

        kl_loss  = self.compute_KL_loss(latent) * config['kl_w']
        rec_loss = self.mse_crit(recons,images) 
        total_loss = rec_loss + kl_loss
        total_loss.backward()
        self.vae_optim.step()

        self.vae_kl_loss    = kl_loss.item()
        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()

        self.encoder_samples = samples.data

        return recons

    def compute_KL_loss(self,distribution):
        mu_2    = torch.pow(distribution['mean'],2)
        sigma_2 = torch.pow(distribution['std'],2)
        return (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()
        #return (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2)).mean()

    def get_latent_visualization(self,image_directory,postfix,images,prior_samples):
        with torch.no_grad():
            recons, latent, samples = self.vae(images)
            start = latent['mean'].clone().detach()

            out_list1 = [recons[0:1]]
            out_list2 = [recons[1:2]]
            out_list3 = [recons[2:3]]
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            recons = torch.cat(out_list,0)
            file_name = '%s/recons_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(recons/2 + 0.5, file_name, nrow=11)

            #######################################
            geners = self.vae.decoder(prior_samples)
            out_list1 = [geners[0:1]]
            out_list2 = [geners[1:2]]
            out_list3 = [geners[2:3]]

            start = prior_samples.clone().detach()
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            geners = torch.cat(out_list,0)
            file_name = '%s/geners_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(geners/2 + 0.5, file_name, nrow=11)

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        vae_name = os.path.join(snapshot_dir, 'vae_%08d.pt' % (iterations + 1))
        torch.save(self.vae.state_dict(), vae_name)

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "vae")
        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])


#trains VAEGAN
class TrainerVAEGAN():
    def __init__(self, config):
        #super(Trainer, self).__init__()
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = VAE(imgconf,config['gen'],config['latent'])
        self.dis = DiscriminatorVAE(config['dis'],imgconf['image_size'],imgconf['image_dim'])
        '''
        disconf = config['dis']
        self.dis = DiscriminatorVAE(disconf['n_downsample'],disconf['n_res'],
                                    imgconf['image_size'],imgconf['image_dim'],
                                    disconf['dim'],disconf['norm'],disconf['activ'],disconf['pad_type'])
        '''
        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)
        self.dis_optim = optim.Adam(self.dis.parameters(),lr=lr)

        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        self.dis_scheduler = get_scheduler(self.dis_optim, config)
        '''
        beta1 = config['beta1']
        beta2 = config['beta2']
        self.vae_optim = optim.Adam(self.vae.parameters(),
                                    lr=lr, betas=(beta1, beta2), weight_decay=config['weight_decay'])
        self.dis_optim = optim.Adam(self.dis.parameters(),
                                    lr=lr, betas=(beta1, beta2), weight_decay=config['weight_decay'])
        '''

        self.mse_crit = nn.MSELoss()
        self.bce_vae  = nn.BCELoss()
        self.bce_dis  = nn.BCELoss()

        '''
        self.vae.apply(weights_init(config['init']))
        self.dis.apply(weights_init('gaussian'))
        '''

    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,config):
        if not config['vae_adv_full']:
            return self.__update_vae_adv_dec(images,config)
        else:
            return self.__update_vae_adv_all(images,config)


    def __update_vae_adv_dec(self,images,config):
        #passes adv grad to decoder
        self.vae_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        # vae part update
        recons, latent, samples = self.vae(inputs)

        kl_loss  = self.compute_KL_loss(latent) * config['kl_w']
        rec_loss = self.mse_crit(recons,images) 
        total_loss = rec_loss + kl_loss
        total_loss.backward()
        self.vae_optim.step()

        # gan part update
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        #pass gan gradient only to decoder
        samples       = samples.detach()
        prior_samples = self.vae.prior.sample_prior(batch_size,images.device)
        all_samples   = torch.cat([samples,prior_samples],0)

        geners   = self.vae.decoder(all_samples)
        fake_out = self.dis(geners).view(geners.size(0),-1)

        adv_loss = self.dis.calc_gen_loss(fake_out) * config['adv_w']
        adv_loss.backward() 
        self.vae_optim.step()

        #not added adv to total loss
        self.vae_kl_loss    = kl_loss.item()
        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()
        self.vae_adv_loss   = adv_loss.item()

        self.encoder_samples = samples.data

        return recons

    def __update_vae_adv_all(self,images,config):
        #passes adv grad through vae
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        # vae part
        recons, latent, samples = self.vae(inputs)
        kl_loss  = self.compute_KL_loss(latent) * config['kl_w']
        rec_loss = self.mse_crit(recons,images) 

        #gan part
        prior_samples = self.vae.prior.sample_prior(batch_size,images.device)
        geners        = self.vae.decoder(prior_samples)

        fake      = torch.cat([geners,recons],0)
        fake_out  = self.dis(fake).view(fake.size(0),-1)
        adv_loss  = self.dis.calc_gen_loss(fake_out) * config['adv_w']

        total_loss = rec_loss + kl_loss + adv_loss

        total_loss.backward()
        self.vae_optim.step()

        self.vae_kl_loss    = kl_loss.item()
        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()
        self.vae_adv_loss   = adv_loss.item()

        self.encoder_samples = samples.data

        return recons

    def compute_KL_loss(self,distribution):
        mu_2    = torch.pow(distribution['mean'],2)
        sigma_2 = torch.pow(distribution['std'],2)
        return (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()
        #return (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(0)).mean()

    def update_dis(self,images,config):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        batch_size = images.size(0)
        inputs = images

        with torch.no_grad():
            recons, _, _ = self.vae(inputs)
            prior_samples = self.vae.prior.sample_prior(batch_size)
            geners = self.vae.decoder(prior_samples) 

        recons = recons.detach()
        geners = geners.detach()
        fake   = torch.cat([recons,geners],0)

        fake_out = self.dis(fake)
        real_out = self.dis(images)
        '''
        fake_loss = self.bce_dis(fake_out,fake_l)
        real_loss = self.bce_dis(real_out,real_l)
        dis_total_loss = self.dis.calc_dis_loss(fake_out, real_out) * config['adv_w'] * (fake_loss + real_loss) 
        '''
        dis_total_loss, fake_loss, real_loss = self.dis.calc_dis_loss(fake_out, real_out)
        dis_total_loss *= config['adv_w']

        dis_total_loss.backward()
        self.dis_optim.step()

        self.dis_fake_loss  = fake_loss.item()
        self.dis_real_loss  = real_loss.item()
        self.dis_total_loss = dis_total_loss.item()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        vae_name = os.path.join(snapshot_dir, 'vae_%08d.pt' % (iterations + 1))
        torch.save(self.vae.state_dict(), vae_name)

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "vae")
        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)

    def get_latent_visualization(self,image_directory,postfix,images,prior_samples):
        with torch.no_grad():
            recons, latent, samples = self.vae(images)
            start = latent['mean'].clone().detach()

            out_list1 = [recons[0:1]]
            out_list2 = [recons[1:2]]
            out_list3 = [recons[2:3]]
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            recons = torch.cat(out_list,0)
            file_name = '%s/recons_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(recons/2 + 0.5, file_name, nrow=11)

            #######################################
            geners = self.vae.decoder(prior_samples)
            out_list1 = [geners[0:1]]
            out_list2 = [geners[1:2]]
            out_list3 = [geners[2:3]]

            start = prior_samples.clone().detach()
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            geners = torch.cat(out_list,0)
            file_name = '%s/geners_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(geners/2 + 0.5, file_name, nrow=11)


#trains VAEGAN with MI maximization
class TrainerInfoVAEGAN():
    def __init__(self, config):
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = VAE(imgconf,config['gen'],config['latent'])
        self.dis = DiscriminatorInfoVAE(config['dis'],imgconf['image_size'],imgconf['image_dim'],config['latent'])

        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)
        self.dis_optim = optim.Adam(self.dis.parameters(),lr=lr)

        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        self.dis_scheduler = get_scheduler(self.dis_optim, config)

        self.mse_crit = nn.MSELoss()
        self.bce_vae  = nn.BCELoss()
        self.bce_dis  = nn.BCELoss()


    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,config):
        #passes adv grad through full vae
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        ##### vae part
        recons, latent, samples = self.vae(inputs)

        kl_loss  = self.compute_KL_loss(latent) * config['kl_w']
        rec_loss = self.mse_crit(recons,images) 

        ##### gan part
        prior_samples = self.vae.prior.sample_prior(batch_size,images.device)
        geners        = self.vae.decoder(prior_samples)

        fake              = torch.cat([geners,recons],0)
        fake_out, q_dist  = self.dis(fake)
        adv_loss          = self.dis.calc_gen_loss(fake_out) * config['adv_w']

        #####need to separate prior likelihood from latent for full mi loss
        ##### info part
        inf_dim  = config['latent']['inform_dim']
        inf_code = torch.cat([prior_samples[:,:inf_dim],samples[:,:inf_dim]],0)

        mi_loss = self.compute_mi(inf_code,q_dist) * config['inf_w']

        total_loss = rec_loss + kl_loss + adv_loss + mi_loss

        total_loss.backward()
        self.vae_optim.step()

        self.vae_kl_loss    = kl_loss.item()
        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()
        self.vae_adv_loss   = adv_loss.item()
        self.vae_inf_loss   = mi_loss.item()

        self.encoder_samples = samples.data

        return recons

    def compute_mi(self, samples, q_dist_raw):
        #so far computes only entropy of Q(c|X)
        q_dist  = self.vae.prior.activate(q_dist_raw)
        qx_li    = self.vae.prior.log_li(samples, q_dist)
        qx_ent   = torch.mean(-qx_li)

        return qx_ent

    def compute_KL_loss(self,distribution):
        mu_2    = torch.pow(distribution['mean'],2)
        sigma_2 = torch.pow(distribution['std'],2)
        return (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()

    def update_dis(self,images,config):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        batch_size = images.size(0)
        inputs = images

        with torch.no_grad():
            recons, latent, samples = self.vae(inputs)
            samples = samples.detach()

            prior_samples = self.vae.prior.sample_prior(batch_size).detach()
            geners = self.vae.decoder(prior_samples) 

        recons = recons.detach()
        geners = geners.detach()
        fake   = torch.cat([geners,recons],0)

        fake_out, q_dist = self.dis(fake)
        real_out, _        = self.dis(images)

        inf_dim  = config['latent']['inform_dim']
        inf_code = torch.cat([prior_samples[:,:inf_dim],samples[:,:inf_dim]],0)

        mi_loss = self.compute_mi(inf_code, q_dist) * config['inf_w']

        dis_loss, fake_loss, real_loss = self.dis.calc_dis_loss(fake_out, real_out)
        dis_loss *= config['adv_w']

        dis_total_loss = dis_loss + mi_loss

        dis_total_loss.backward()
        self.dis_optim.step()

        self.dis_fake_loss  = fake_loss.item()
        self.dis_real_loss  = real_loss.item()
        self.dis_loss       = dis_loss.item()
        self.dis_mi_loss    = mi_loss.item()
        self.dis_total_loss = dis_total_loss.item()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        vae_name = os.path.join(snapshot_dir, 'vae_%08d.pt' % (iterations + 1))
        torch.save(self.vae.state_dict(), vae_name)

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "vae")
        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)

    def get_latent_visualization(self,image_directory,postfix,images,prior_samples):
        with torch.no_grad():
            recons, latent, samples = self.vae(images)
            start = latent['mean'].clone().detach()

            out_list1 = [recons[0:1]]
            out_list2 = [recons[1:2]]
            out_list3 = [recons[2:3]]
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            recons = torch.cat(out_list,0)
            file_name = '%s/recons_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(recons/2 + 0.5, file_name, nrow=11)

            #######################################
            geners = self.vae.decoder(prior_samples)
            out_list1 = [geners[0:1]]
            out_list2 = [geners[1:2]]
            out_list3 = [geners[2:3]]

            start = prior_samples.clone().detach()
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            geners = torch.cat(out_list,0)
            file_name = '%s/geners_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(geners/2 + 0.5, file_name, nrow=11)


#trains VAE with gaussian priror and MI maximization
class TrainerInfoVAE():
    def __init__(self, config):
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = VAE(imgconf,config['gen'],config['latent'])
        self.dis = QNet(config['dis'],imgconf['image_size'],imgconf['image_dim'],config['latent'])

        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)
        self.dis_optim = optim.Adam(self.dis.parameters(),lr=lr)

        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        self.dis_scheduler = get_scheduler(self.dis_optim, config)

        self.mse_crit = nn.MSELoss()
        self.bce_vae  = nn.BCELoss()
        self.bce_dis  = nn.BCELoss()


    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,config):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        ##### vae part
        recons, latent, samples = self.vae(inputs)

        kl_loss  = self.compute_KL_loss(latent) * config['kl_w']
        rec_loss = self.mse_crit(recons,images) 

        ##### gan part
        prior_samples = self.vae.prior.sample_prior(batch_size,images.device)
        geners        = self.vae.decoder(prior_samples)

        fake    = torch.cat([geners,recons],0)
        q_dist  = self.dis(fake)

        #####need to separate prior likelihood from latent for full mi loss
        ##### info part
        inf_dim  = config['latent']['inform_dim']
        inf_code = torch.cat([prior_samples[:,:inf_dim],samples[:,:inf_dim]],0)

        mi_loss = self.compute_mi(inf_code, q_dist) * config['inf_w']

        total_loss = rec_loss + kl_loss + mi_loss

        total_loss.backward()
        self.vae_optim.step()

        self.vae_kl_loss    = kl_loss.item()
        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()
        self.vae_inf_loss   = mi_loss.item()

        self.encoder_samples = samples.data

        return recons

    def compute_mi(self, samples, q_dist_raw):
        #so far computes only entropy of Q(c|X)
        q_dist  = self.vae.prior.activate(q_dist_raw)
        qx_li    = self.vae.prior.log_li(samples, q_dist)
        qx_ent   = torch.mean(-qx_li)

        return qx_ent

    def compute_KL_loss(self,distribution):
        mu_2    = torch.pow(distribution['mean'],2)
        sigma_2 = torch.pow(distribution['std'],2)
        return (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()

    def update_dis(self,images,config):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        batch_size = images.size(0)
        inputs = images

        with torch.no_grad():
            recons, latent, samples = self.vae(inputs)
            samples = samples.detach()

            prior_samples = self.vae.prior.sample_prior(batch_size).detach()
            geners = self.vae.decoder(prior_samples) 

        recons = recons.detach()
        geners = geners.detach()
        fake   = torch.cat([geners,recons],0)

        q_dist = self.dis(fake)

        inf_dim  = config['latent']['inform_dim']
        inf_code = torch.cat([prior_samples[:,:inf_dim],samples[:,:inf_dim]],0)

        mi_loss = self.compute_mi(inf_code, q_dist)

        dis_total_loss = mi_loss

        dis_total_loss.backward()
        self.dis_optim.step()

        self.dis_mi_loss = mi_loss.item()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        vae_name = os.path.join(snapshot_dir, 'vae_%08d.pt' % (iterations + 1))
        torch.save(self.vae.state_dict(), vae_name)

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "vae")
        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)

    def get_latent_visualization(self,image_directory,postfix,images,prior_samples):
        with torch.no_grad():
            recons, latent, samples = self.vae(images)
            start = latent['mean'].clone().detach()

            out_list1 = [recons[0:1]]
            out_list2 = [recons[1:2]]
            out_list3 = [recons[2:3]]
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            recons = torch.cat(out_list,0)
            file_name = '%s/recons_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(recons/2 + 0.5, file_name, nrow=11)

            #######################################
            geners = self.vae.decoder(prior_samples)
            out_list1 = [geners[0:1]]
            out_list2 = [geners[1:2]]
            out_list3 = [geners[2:3]]

            start = prior_samples.clone().detach()
            for i in range(10):
                start.data[:3,0] = -5. + i
                out = self.vae.decoder(start)
                out_list1.append(out[0:1])
                out_list2.append(out[1:2])
                out_list3.append(out[2:3])

            out_list = out_list1 + out_list2 + out_list3
            geners = torch.cat(out_list,0)
            file_name = '%s/geners_intp%s.jpg' % (image_directory, postfix)
            vutils.save_image(geners/2 + 0.5, file_name, nrow=11)


#trains VAE with categotical latent variable 
class TrainerCatVAE():
    def __init__(self, config):
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = CatVAE(imgconf,config['gen'],config['latent'])

        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)

        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        
        self.mse_crit = nn.MSELoss()

    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,iteration):
        self.vae_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        # vae part update
        tempr = self.compute_temperature(iteration)
        recons, samples, categorical_dis_act, continious_dis_act = self.vae(inputs, tempr)

        kl_cont, kl_catg  = self.compute_KL_loss(continious_dis_act, categorical_dis_act)
        kl_loss = (kl_cont + kl_catg) * self.config['kl_w']

        rec_loss = self.mse_crit(recons,images) 
        #total loss, minimize rec_err, KLdis, KLcont
        total_loss = rec_loss + kl_loss

        total_loss.backward()
        self.vae_optim.step()

        #extract losses for logging
        self.vae_kl_loss             = kl_loss.item()
        self.vae_kl_continious_loss  = kl_cont.item()
        self.vae_kl_categorical_loss = kl_catg.item()

        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()

        self.encoder_samples = categorical_dis_act['prob'].data

        return recons

    def compute_KL_loss(self, distribution_cont, distribution_catg):
        #continious kl, maybe moove to Gaussian class
        mu_2    = torch.pow(distribution_cont['mean'],2)
        sigma_2 = torch.pow(distribution_cont['std'],2)
        kl_cont = (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()

        #discrete kl
        catg_prior_info = self.vae.prior_catg.prior_dist_info(distribution_catg['prob'].size(0))
        kl_catg = self.vae.prior_catg.compute_KL(distribution_catg, catg_prior_info).mean()

        return kl_cont, kl_catg

    def compute_temperature(self,iteration):
        #computes adaptive temperature for gumbel trick smoothing
        tempr_config = self.config['temperature']

        start    = tempr_config['start']
        minim    = tempr_config['minim']
        ann_rate = tempr_config['ann_rate']
        iter_dec = tempr_config['iter_dec']

        temp = start
        if (iteration % iter_dec == 1):
            temp = np.maximum(start*np.exp(-ann_rate*iteration),minim)

        return temp

    def save(self, snapshot_dir, iterations):
        # Saves VAE
        vae_name = os.path.join(snapshot_dir, 'vae_%08d.pt' % (iterations + 1))
        torch.save(self.vae.state_dict(), vae_name)

    def resume(self, checkpoint_dir, hyperparameters, particular=False, checkpoint=''):
        # Load generators
        # if particular false then loads 
        if not particular:
            last_model_name = get_model_list(checkpoint_dir, "vae")
        else:
            if checkpoint == '':
                sys.exit('Specified checkpoint path is empty')
            last_model_name = os.path.join(checkpoint_dir, checkpoint)

        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)


#trains VAE with categotical latent variable with MI maximization
class TrainerInfoCatVAE():
    def __init__(self, config):
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = CatVAE(imgconf,config['gen'],config['latent'])
        self.dis = CatQNet(config['dis'],imgconf['image_size'],imgconf['image_dim'],config['latent'])

        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)
        self.dis_optim = optim.Adam(self.dis.parameters(),lr=lr)

        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        self.dis_scheduler = get_scheduler(self.dis_optim, config)

        self.mse_crit      = nn.MSELoss()

    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,iteration):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        # vae part update
        tempr = self.compute_temperature(iteration)
        recons, samples, categorical_dis_act, continious_dis_act = self.vae(inputs, tempr)

        rec_loss = self.mse_crit(recons,images) 

        kl_cont, kl_catg  = self.compute_KL_loss(continious_dis_act, categorical_dis_act)
        kl_loss = (kl_cont + kl_catg) * self.config['kl_w']

        ##### Q net
        catg_dim = self.config['latent']['categorical']
        inf_dim  = self.config['latent']['inform_dim'] # dims of gaussian part to maximize MI

        fake    = recons
        q_dist  = self.dis(fake)

        inf_catg_code = samples[:,-catg_dim:]
        q_dist_catg   = q_dist[:,-catg_dim:]
        mi_catg_loss  = self.compute_mi_categorical(inf_catg_code, q_dist_catg) 

        mi_cont_loss = 0 
        if inf_dim != 0:
            inf_cont_code = samples[:,:inf_dim]
            q_dist_cont   = q_dist[:,:inf_dim*2]
            mi_cont_loss  = self.compute_mi_continious(inf_cont_code, q_dist_cont)

        mi_loss = mi_catg_loss*self.config['inf_w_catg'] + mi_cont_loss*self.config['inf_w_cont']

        #total loss, minimize rec_err, KLdis, KLcont, MI_loss
        total_loss = rec_loss + kl_loss + mi_loss

        total_loss.backward()
        self.vae_optim.step()

        #extract losses for logging
        self.vae_kl_loss             = kl_loss.item()
        self.vae_kl_continious_loss  = kl_cont.item()
        self.vae_kl_categorical_loss = kl_catg.item()

        self.vae_mi_loss      = mi_loss.item()
        self.vae_mi_catg_loss = mi_catg_loss.item()
        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()
        if inf_dim != 0:
            self.vae_mi_cont_loss = mi_cont_loss.item()

        self.encoder_samples = categorical_dis_act['prob'].data

        return recons

    def update_dis(self,images,iteration):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        batch_size = images.size(0)
        inputs = images

        tempr = self.compute_temperature(iteration)
        with torch.no_grad():
            recons, samples, categorical_dis_act, continious_dis_act = self.vae(inputs, tempr)
            samples = samples.detach()

        recons = recons.detach()

        catg_dim = self.config['latent']['categorical']
        inf_dim  = self.config['latent']['inform_dim'] # dims of gaussian part to maximize MI

        fake = recons

        q_dist  = self.dis(fake)

        inf_catg_code = samples[:,-catg_dim:]
        q_dist_catg   = q_dist[:,-catg_dim:]
        mi_catg_loss  = self.compute_mi_categorical(inf_catg_code, q_dist_catg) 

        mi_cont_loss = 0 
        if inf_dim != 0:
            inf_cont_code = samples[:,:inf_dim]
            q_dist_cont   = q_dist[:,:inf_dim*2]
            mi_cont_loss  = self.compute_mi_continious(inf_cont_code, q_dist_cont)

        mi_loss = mi_catg_loss + mi_cont_loss

        dis_total_loss = mi_loss

        mi_loss.backward()
        self.dis_optim.step()

        self.dis_mi_catg_loss = mi_catg_loss.item()
        if inf_dim != 0:
            self.dis_mi_cont_loss = mi_cont_loss.item()
        self.dis_mi_loss = mi_loss.item()

    def compute_mi_categorical(self, samples, q_dist_raw):
        # so far computes only entropy of Q(c|X), negetive likelihood, entropy
        q_dist   = self.vae.prior_catg.activate(q_dist_raw)
        qx_li    = self.vae.prior_catg.log_li(samples, q_dist)
        qx_ent   = torch.mean(-qx_li)
        return qx_ent

    def compute_mi_continious(self, samples, q_dist_raw):
        # so far computes only entropy of Q(c|X), negetive likelihood, entropy
        q_dist   = self.vae.prior_cont.activate(q_dist_raw)
        qx_li    = self.vae.prior_cont.log_li(samples, q_dist)
        qx_ent   = torch.mean(-qx_li)
        return qx_ent

    def compute_KL_loss(self, distribution_cont, distribution_catg):
        #continious kl, maybe moove to Gaussian class
        mu_2    = torch.pow(distribution_cont['mean'],2)
        sigma_2 = torch.pow(distribution_cont['std'],2)
        kl_cont = (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()

        #discrete kl
        catg_prior_info = self.vae.prior_catg.prior_dist_info(distribution_catg['prob'].size(0))
        kl_catg         = self.vae.prior_catg.compute_KL(distribution_catg, catg_prior_info).mean()

        return kl_cont, kl_catg

    def compute_temperature(self,iteration):
        #computes adaptive temperature for gumbel trick smoothing
        tempr_config = self.config['temperature']

        start    = tempr_config['start']
        minim    = tempr_config['minim']
        ann_rate = tempr_config['ann_rate']
        iter_dec = tempr_config['iter_dec']

        temp = start
        if (iteration % iter_dec == 1):
            temp = np.maximum(start*np.exp(-ann_rate*iteration),minim)

        return temp

    def save(self, snapshot_dir, iterations):
        # Saves VAE
        vae_name = os.path.join(snapshot_dir, 'vae_%08d.pt' % (iterations + 1))
        torch.save(self.vae.state_dict(), vae_name)

    def resume(self, checkpoint_dir, hyperparameters, particular=False, checkpoint=''):
        # Load generators
        # if particular false then loads 
        if not particular:
            last_model_name = get_model_list(checkpoint_dir, "vae")
        else:
            if checkpoint == '':
                sys.exit('Specified checkpoint path is empty')
            last_model_name = os.path.join(checkpoint_dir, checkpoint)

        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)


#trains VAE with categotical latent variable MI maximization only for Q net
class TrainerCatVAEInfoEvaluation():
    def __init__(self, config):
        self.config = config
        lr = config['lr']
        # Initiate the networks
        imgconf = config['image']
        self.vae = CatVAE(imgconf,config['gen'],config['latent'])
        self.dis = CatQNet(config['dis'],imgconf['image_size'],imgconf['image_dim'],config['latent'])

        self.vae_optim = optim.Adam(self.vae.parameters(),lr=lr)
        self.dis_optim = optim.Adam(self.dis.parameters(),lr=lr)

        self.vae_scheduler = get_scheduler(self.vae_optim, config)
        self.dis_scheduler = get_scheduler(self.dis_optim, config)
        
        self.mse_crit = nn.MSELoss()

    def update_learning_rate(self):
        if self.vae_scheduler is not None:
            self.vae_scheduler.step()

    def cuda(self,device=0):
        members = [attr for attr in dir(self) if isinstance(getattr(self, attr),torch.nn.Module)]
        for m in members:
            getattr(self, m).cuda(device)

    def update_vae(self,images,iteration):
        self.vae_optim.zero_grad() 

        inputs = images
        batch_size = images.size(0)

        # vae part update
        tempr = self.compute_temperature(iteration)
        recons, samples, categorical_dis_act, continious_dis_act = self.vae(inputs, tempr)

        kl_cont, kl_catg  = self.compute_KL_loss(continious_dis_act, categorical_dis_act)
        kl_loss = (kl_cont + kl_catg) * self.config['kl_w']

        rec_loss = self.mse_crit(recons,images) 
        #total loss, minimize rec_err, KLdis, KLcont
        total_loss = rec_loss + kl_loss

        total_loss.backward()
        self.vae_optim.step()

        #extract losses for logging
        self.vae_kl_loss             = kl_loss.item()
        self.vae_kl_continious_loss  = kl_cont.item()
        self.vae_kl_categorical_loss = kl_catg.item()

        self.vae_rec_loss   = rec_loss.item()
        self.vae_total_loss = total_loss.item()

        self.encoder_samples = categorical_dis_act['prob'].data

        return recons

    def compute_KL_loss(self, distribution_cont, distribution_catg):
        #continious kl, maybe moove to Gaussian class
        mu_2    = torch.pow(distribution_cont['mean'],2)
        sigma_2 = torch.pow(distribution_cont['std'],2)
        kl_cont = (-0.5 * (1 + torch.log(sigma_2) - mu_2 - sigma_2).sum(1)).mean()

        #discrete kl
        catg_prior_info = self.vae.prior_catg.prior_dist_info(distribution_catg['prob'].size(0))
        kl_catg = self.vae.prior_catg.compute_KL(distribution_catg, catg_prior_info).mean()

        return kl_cont, kl_catg

    def update_dis(self,images,iteration):
        self.vae_optim.zero_grad() 
        self.dis_optim.zero_grad() 

        batch_size = images.size(0)
        inputs = images

        tempr = self.compute_temperature(iteration)
        with torch.no_grad():
            recons, samples, categorical_dis_act, continious_dis_act = self.vae(inputs, tempr)
            samples = samples.detach()

            prior_samples = self.vae.sample_full_prior(batch_size).detach()
            geners = self.vae.decoder(prior_samples) 

        recons = recons.detach()
        geners = geners.detach()

        catg_dim = self.config['latent']['categorical']
        if self.config['use_gen_info']:
            inf_catg_code = torch.cat([prior_samples[:,-catg_dim:],samples[:,-catg_dim:]],0)
            fake = torch.cat([geners,recons],0)
        else:
            inf_catg_code = samples[:,-catg_dim:]
            fake = recons

        q_dist  = self.dis(fake)

        mi_catg_loss  = self.compute_mi_categorical(inf_catg_code, q_dist) 
        mi_cont_loss = 0 #to be implemented
        mi_loss = mi_catg_loss + mi_cont_loss

        dis_total_loss = mi_loss

        mi_loss.backward()
        self.dis_optim.step()

        self.dis_mi_catg_loss = mi_catg_loss.item()
        #self.dis_mi_cont_loss = mi_cont_loss.item() #to be implemented
        self.dis_mi_loss = mi_loss.item()


    def compute_mi_categorical(self, samples, q_dist_raw):
        #so far computes only entropy of Q(c|X), negetive likelihood
        q_dist  = self.vae.prior_catg.activate(q_dist_raw)
        qx_li    = self.vae.prior_catg.log_li(samples, q_dist)
        qx_ent   = torch.mean(-qx_li)

        return qx_ent#prior_ent - qx_ent

    def compute_temperature(self,iteration):
        #computes adaptive temperature for gumbel trick smoothing
        tempr_config = self.config['temperature']

        start    = tempr_config['start']
        minim    = tempr_config['minim']
        ann_rate = tempr_config['ann_rate']
        iter_dec = tempr_config['iter_dec']

        temp = start
        if (iteration % iter_dec == 1):
            temp = np.maximum(start*np.exp(-ann_rate*iteration),minim)

        return temp


    def resume(self, checkpoint_dir, hyperparameters, particular=False, checkpoint=''):
        # Load generators
        # if particular false then loads 
        if not particular:
            last_model_name = get_model_list(checkpoint_dir, "vae")
        else:
            if checkpoint == '':
                sys.exit('Specified checkpoint path is empty')
            last_model_name = os.path.join(checkpoint_dir, checkpoint)

        state_dict = torch.load(last_model_name)
        self.vae.load_state_dict(state_dict)
