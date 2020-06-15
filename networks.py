import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from distributions import Gaussian, Categorical, Gumbel

##################################################################################
# Encoder for Gaussian VAE
##################################################################################

class Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, n_mlp, input_size, input_dim, dim, mlp_dim,
                 latent_dim, norm='bn', activ='lrelu', pad_type='zero'):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # downsampling blocks
        conv_blk = []
        conv_blk += [Conv2dBlock(input_dim,dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
        for i in range(n_downsample-1):
            conv_blk += [Conv2dBlock(dim,2*dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
            dim *= 2
        # residual blocks
        if n_res > 0:
            conv_blk += [ResBlocks(n_res, dim,norm=norm,activation=activ,pad_type=pad_type)]
        self.conv_blk = nn.Sequential(*conv_blk)

        # Linear blocks 
        self.mlp = MLP(dim*(input_size//(2**(n_downsample)))**2, latent_dim*2, mlp_dim, n_mlp, norm=norm, activ=activ)

    def forward(self, inp):
        b_size = inp.size(0)
        conv_out   = self.conv_blk(inp)
        conv_out   = conv_out.view(b_size,-1)

        mlp_out    = self.mlp(conv_out)
        return mlp_out

##################################################################################
# Decoder network for all VAEs
##################################################################################

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, n_mlp, latent_dim, mlp_dim, conv_inp_size, dim,
                 output_dim, norm='bn', activ='lrelu', pad_type='zero'):
        super(Decoder, self).__init__()
        self.conv_inp_size = conv_inp_size
        self.dim = dim

        self.mlp = MLP(latent_dim, (conv_inp_size**2)*dim, mlp_dim, n_mlp, norm=norm, activ=activ)

        deconv_blk = []
        if n_res > 0:
            deconv_blk += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks used to make by upsampling module and then siple convolution
        for i in range(n_upsample - 1):
            deconv_blk += [Conv2dTransposeBlock(dim, dim // 2, 4, 2, 1, norm=norm, activation=activ)]
            dim //= 2
        deconv_blk += [Conv2dTransposeBlock(dim, output_dim, 4, 2, 1, norm='none', activation='tanh')]

        self.deconv_blk = nn.Sequential(*deconv_blk)

    def forward(self, x):
        b_size = x.size(0)

        mlp_out = self.mlp(x)
        mlp_out = mlp_out.view(b_size,self.dim,self.conv_inp_size,self.conv_inp_size)

        deconv_blk_out = self.deconv_blk(mlp_out)
        return deconv_blk_out

##################################################################################
# Joint categorica and gauissian VAE encoder
##################################################################################

class CatEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, n_mlp, input_size, input_dim, dim, mlp_dim,
                 latent, norm='bn', activ='lrelu', pad_type='zero'):
        super(CatEncoder, self).__init__()

        continious_dim  = latent['continious']
        categorical_dim = latent['categorical']

        # downsampling blocks
        conv_blk = []
        conv_blk += [Conv2dBlock(input_dim,dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
        for i in range(n_downsample-1):
            conv_blk += [Conv2dBlock(dim,2*dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
            dim *= 2
        # residual blocks
        if n_res > 0:
            conv_blk += [ResBlocks(n_res, dim,norm=norm,activation=activ,pad_type=pad_type)]
        self.conv_blk = nn.Sequential(*conv_blk)

        # Linear blocks 
        self.mlp = MLP(dim*(input_size//(2**(n_downsample)))**2, 
                       continious_dim*2 + categorical_dim, 
                       mlp_dim, n_mlp, 
                       norm=norm, activ=activ)

    def forward(self, inp):
        b_size = inp.size(0)

        conv_out   = self.conv_blk(inp)
        conv_out   = conv_out.view(b_size,-1)

        mlp_out    = self.mlp(conv_out)
        return mlp_out

##################################################################################
# Gaussian VAE
##################################################################################

class VAE(nn.Module):
    def __init__(self,img_params,model_params,latent_params):
        super(VAE, self).__init__()
        image_dim  = img_params['image_dim']
        image_size = img_params['image_size']

        n_downsample = model_params['n_downsample']
        dim      = model_params['dim']
        n_res    = model_params['n_res']
        norm     = model_params['norm']
        activ    = model_params['activ']
        pad_type = model_params['pad_type']
        n_mlp    = model_params['n_mlp']
        mlp_dim  = model_params['mlp_dim']

        self.latent_dim = latent_params['latent_dim']
        self.prior = Gaussian(self.latent_dim)

        self.encoder = Encoder(n_downsample,n_res,n_mlp,image_size,image_dim,dim,mlp_dim,
                               self.latent_dim,norm,activ,pad_type)

        conv_inp_size = image_size // (2**n_downsample)
        self.decoder = Decoder(n_downsample,n_res,n_mlp,self.latent_dim,mlp_dim,conv_inp_size,
                               dim,image_dim,norm,activ,pad_type)

    def forward(self,x):
        latent_distr = self.encoder(x)
        latent_distr = self.prior.activate(latent_distr)
        samples = self.prior.sample(latent_distr)
        return self.decoder(samples),latent_distr,samples


##################################################################################
# Joint categorica and gauissian VAE
##################################################################################

class CatVAE(nn.Module):
    # Auto-encoder architecture
    def __init__(self,img_params,model_params,latent_params):
        super(CatVAE, self).__init__()
        image_dim  = img_params['image_dim']
        image_size = img_params['image_size']

        n_downsample = model_params['n_downsample']
        dim      = model_params['dim']
        n_res    = model_params['n_res']
        norm     = model_params['norm']
        activ    = model_params['activ']
        pad_type = model_params['pad_type']
        n_mlp    = model_params['n_mlp']
        mlp_dim  = model_params['mlp_dim']

        self.continious_dim = latent_params['continious']
        self.prior_cont = Gaussian(self.continious_dim)

        self.categorical_dim = latent_params['categorical']
        self.prior_catg = Categorical(self.categorical_dim)
        self.gumbel     = Gumbel(self.categorical_dim)

        self.encoder = CatEncoder(n_downsample,n_res,n_mlp,image_size,image_dim,dim,mlp_dim,
                                  latent_params,norm,activ,pad_type)

        conv_inp_size = image_size // (2**n_downsample)
        decoder_inp_dim = self.continious_dim + self.categorical_dim
        self.decoder = Decoder(n_downsample,n_res,n_mlp,decoder_inp_dim,mlp_dim,conv_inp_size,
                               dim,image_dim,norm,activ,pad_type)

    def forward(self, x, tempr):
        latent_distr = self.encoder(x)

        #categorical distr
        categorical_distr     = latent_distr[:,-self.categorical_dim:]
        categorical_distr_act = self.prior_catg.activate(categorical_distr)# need for KL

        catg_samples = self.gumbel.gumbel_softmax_sample(categorical_distr,tempr) # categotical sampling, reconstruction

        #continious distr
        continious_distr     = latent_distr[:,:-self.categorical_dim]
        continious_distr_act = self.prior_cont.activate(continious_distr)
        cont_samples         = self.prior_cont.sample(continious_distr_act)

        #create full latent code
        full_samples = torch.cat([cont_samples,catg_samples],1)

        recons = self.decoder(full_samples)

        return recons, full_samples, categorical_distr_act, continious_distr_act
        
    def encode_decode(self, x, tempr=0.4, hard_catg=True):
        latent_distr = self.encoder(x)

        #categorical distr stuff
        categorical_distr = latent_distr[:,-self.categorical_dim:]
        if hard_catg:
            #just make one hot vector
            catg_samples = self.prior_catg.logits_to_onehot(categorical_distr)
        else:
            #make smoothed one hot by softmax
            catg_samples = self.prior_catg.activate(categorical_distr)['prob']

        #continious distr stuff
        continious_distr     = latent_distr[:,:-self.categorical_dim]
        continious_distr_act = self.prior_cont.activate(continious_distr)
        cont_samples         = continious_distr_act['mean']

        #create full latent code
        full_samples = torch.cat([cont_samples,catg_samples],1)
        recons = self.decoder(full_samples)

        return recons, full_samples#, categorical_distr_act, continious_distr_act

    def sample_full_prior(self, batch_size, device='cuda:0'):
        cont_samples = self.prior_cont.sample_prior(batch_size, device=device)
        catg_samples = self.prior_catg.sample_prior(batch_size, device=device)
        full_samples = torch.cat([cont_samples,catg_samples],1)
        return full_samples

##################################################################################
# Discriminator foe VAEGAN experiments
##################################################################################

class DiscriminatorVAE(nn.Module):
    def __init__(self, disconf, input_size, input_dim):
        super(DiscriminatorVAE, self).__init__()
        n_downsample  = disconf['n_downsample']
        n_res         = disconf['n_res']
        dim           = disconf['dim']
        norm          = disconf['norm']
        activ         = disconf['activ']
        pad_type      = disconf['pad_type']
        self.gan_type = disconf['gan_type']
        #self.latent_dim = latent_dim
        # downsampling blocks
        conv_blk = []
        conv_blk += [Conv2dBlock(input_dim,dim,4,2,1,norm='none',activation=activ,pad_type=pad_type)]
        for i in range(n_downsample-1):
            conv_blk += [Conv2dBlock(dim,2*dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
            dim *= 2
        # residual blocks
        if n_res > 0:
            conv_blk += [ResBlocks(n_res, dim,norm=norm,activation=activ,pad_type=pad_type)]

        conv_blk += [Conv2dBlock(dim, 1, input_size//(2**(n_downsample)),1,0, norm='none', activation='none')]
        self.conv_blk = nn.Sequential(*conv_blk)

        self.bce  = nn.BCELoss()
        self.mse  = nn.MSELoss()
        self.sigm = nn.Sigmoid()

    def forward(self, inp):
        b_size = inp.size(0)
        conv_out   = self.conv_blk(inp)
        return conv_out.view(b_size,-1)

    def calc_dis_loss(self, fake_out, real_out):
        # calculate the loss to train D
        all0 = torch.zeros_like(fake_out)
        all1 = torch.ones_like(real_out)
        if self.gan_type == 'lsgan':
            fake_loss = self.mse(fake_out,all0)
            real_loss = self.mse(real_out,all1)
        elif self.gan_type == 'nsgan':
            all1 = all1 - 0.1
            fake_loss = self.bce(self.sigm(fake_out), all0) 
            real_loss = self.bce(self.sigm(real_out), all1) 
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = fake_loss + real_loss
        return loss, fake_loss, real_loss

    def calc_gen_loss(self, fake_out):
        all1 = torch.ones_like(fake_out)
        if self.gan_type == 'lsgan':
            loss = self.mse(fake_out, all1) # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = all1 - 0.1
            loss = self.bce(self.sigm(fake_out), all1)
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Discriminator foe InfoVAEGAN experiments
##################################################################################

class DiscriminatorInfoVAE(nn.Module):
    def __init__(self, disconf, input_size, input_dim, latentconf):
        super(DiscriminatorInfoVAE, self).__init__()
        n_downsample  = disconf['n_downsample']
        n_res         = disconf['n_res']
        dim           = disconf['dim']
        norm          = disconf['norm']
        activ         = disconf['activ']
        pad_type      = disconf['pad_type']
        self.gan_type = disconf['gan_type']

        latent_inf_dim = latentconf['inform_dim']
        # downsampling blocks
        conv_blk = []
        conv_blk += [Conv2dBlock(input_dim,dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
        for i in range(n_downsample-1):
            conv_blk += [Conv2dBlock(dim,2*dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
            dim *= 2
        # residual blocks
        if n_res > 0:
            conv_blk += [ResBlocks(n_res, dim,norm=norm,activation=activ,pad_type=pad_type)]

        self.main_conv_blk = nn.Sequential(*conv_blk)

        self.dis_out = Conv2dBlock(dim, 1, input_size//(2**(n_downsample)),1,0, norm='none', activation='none')

        self.inf_out = nn.Sequential(Conv2dBlock(dim, 128, input_size//(2**(n_downsample)),1,0, norm=norm, activation=activ),
                                     Conv2dBlock(128,latent_inf_dim*2,1,1,0,norm='none',activation='none'))

        self.bce  = nn.BCELoss()
        self.mse  = nn.MSELoss()
        self.sigm = nn.Sigmoid()

    def forward(self, inp):
        b_size = inp.size(0)
        conv_out   = self.main_conv_blk(inp)

        d_out = self.dis_out(conv_out).view(b_size,-1)
        i_out = self.inf_out(conv_out).view(b_size,-1)
        return d_out, i_out 

    def calc_dis_loss(self, fake_out, real_out):
        # calculate the loss to train D
        all0 = torch.zeros_like(fake_out)
        all1 = torch.ones_like(real_out)
        if self.gan_type == 'lsgan':
            fake_loss = self.mse(fake_out,all0)
            real_loss = self.mse(real_out,all1)
        elif self.gan_type == 'nsgan':
            all1 = all1 - 0.1
            fake_loss = self.bce(self.sigm(fake_out), all0) 
            real_loss = self.bce(self.sigm(real_out), all1) 
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = fake_loss + real_loss
        return loss, fake_loss, real_loss

    def calc_gen_loss(self, fake_out):
        all1 = torch.ones_like(fake_out)
        if self.gan_type == 'lsgan':
            loss = self.mse(fake_out, all1) # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = all1 - 0.1
            loss = self.bce(self.sigm(fake_out), all1)
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Network for modelling auxiliary distribution Q (infers codes from decoded imgs)
##################################################################################

class QNet(nn.Module):
    def __init__(self, qnetconf, input_size, input_dim, latentconf):
        super(QNet, self).__init__()
        n_downsample  = qnetconf['n_downsample']
        n_res         = qnetconf['n_res']
        dim           = qnetconf['dim']
        norm          = qnetconf['norm']
        activ         = qnetconf['activ']
        pad_type      = qnetconf['pad_type']
        latent_inf_dim = latentconf['inform_dim']
        #self.latent_dim = latent_dim
        conv_blk = []
        conv_blk += [Conv2dBlock(input_dim,dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
        for i in range(n_downsample-1):
            conv_blk += [Conv2dBlock(dim,2*dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
            dim *= 2
        # residual blocks
        if n_res > 0:
            conv_blk += [ResBlocks(n_res, dim,norm=norm,activation=activ,pad_type=pad_type)]

        conv_blk += [Conv2dBlock(dim, 128, input_size//(2**(n_downsample)),1,0, norm=norm, activation=activ)]
        conv_blk += [Conv2dBlock(128,latent_inf_dim*2,1,1,0,norm='none',activation='none')]

        self.conv_blk = nn.Sequential(*conv_blk)

    def forward(self, inp):
        b_size = inp.size(0)
        conv_out   = self.conv_blk(inp)
        return conv_out.view(b_size,-1)

##################################################################################
# Network for modelling auxiliary distribution Q (infers codes from decoded imgs)
# Joint cat/gaus VAE model
##################################################################################

class CatQNet(nn.Module):
    def __init__(self, qnetconf, input_size, input_dim, latentconf):
        super(CatQNet, self).__init__()
        n_downsample  = qnetconf['n_downsample']
        n_res         = qnetconf['n_res']
        dim           = qnetconf['dim']
        norm          = qnetconf['norm']
        activ         = qnetconf['activ']
        pad_type      = qnetconf['pad_type']

        latent_catg_dim     = latentconf['categorical']
        latent_inf_cont_dim = latentconf['inform_dim']

        output_dim = latent_inf_cont_dim * 2 + latent_catg_dim

        conv_blk = []
        conv_blk += [Conv2dBlock(input_dim,dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
        for i in range(n_downsample-1):
            conv_blk += [Conv2dBlock(dim,2*dim,4,2,1,norm=norm,activation=activ,pad_type=pad_type)]
            dim *= 2
        # residual blocks
        if n_res > 0:
            conv_blk += [ResBlocks(n_res, dim,norm=norm,activation=activ,pad_type=pad_type)]

        conv_blk += [Conv2dBlock(dim, 128, input_size//(2**(n_downsample)),1,0, norm=norm, activation=activ)]
        conv_blk += [Conv2dBlock(128,output_dim,1,1,0,norm='none',activation='none')]

        self.conv_blk = nn.Sequential(*conv_blk)

    def forward(self, inp):
        b_size = inp.size(0)
        conv_out   = self.conv_blk(inp)
        return conv_out.view(b_size,-1)



class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu',out_activ='none'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]

        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]

        self.model += [LinearBlock(dim, output_dim, norm='none', activation=out_activ)] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigm':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Conv2dTransposeBlock(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, output_padding=0, activation='lrelu',norm='bn'):
    super(Conv2dTransposeBlock, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(input_dim, output_dim,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding,bias=True)]

    # add normalization
    norm_dim = output_dim
    if norm == 'bn':
        model += [nn.BatchNorm2d(norm_dim)]
    elif norm == 'in':
        model += [nn.InstanceNorm2d(norm_dim)]
    elif norm == 'ln':
        model += [LayerNorm(norm_dim)]
    elif not (norm == 'none' or norm == 'sn'):
        assert 0, "Unsupported normalization: {}".format(norm)

    # initialize activation
    if activation == 'relu':
        model += [nn.ReLU(inplace=True)]
    elif activation == 'lrelu':
        model += [nn.LeakyReLU(0.2, inplace=True)]
    elif activation == 'prelu':
        model += [nn.PReLU()]
    elif activation == 'selu':
        model += [nn.SELU(inplace=True)]
    elif activation == 'tanh':
        model += [nn.Tanh()]
    elif not activation == 'none':
        assert 0, "Unsupported activation: {}".format(activation)

    self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

  def forward(self, x):
    return self.model(x)

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
