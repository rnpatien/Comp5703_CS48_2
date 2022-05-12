import torch
import torch.nn.functional as F

from third_party.gather_layer import GatherLayer
from training.criterion import nt_xent

def gatherFeatures(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features

def supcon_fake(out1, out2, others, temperature, distributed=False):
    if distributed:
        out1 = torch.cat(GatherLayer.apply(out1), dim=0)
        out2 = torch.cat(GatherLayer.apply(out2), dim=0)
        others = torch.cat(GatherLayer.apply(others), dim=0)
    N = out1.size(0)

    _out = [out1, out2, others]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss


def loss_D_fn(P, D, options, images, gen_images, opt_D=None):
    # print ("damage discrim")
    assert images.size(0) == gen_images.size(0)
    gen_images = gen_images.detach()
    N = images.size(0)

    inputs_1 = P.augment_fn(images)
    inputs_2 = P.augment_fn(images)
    inputs_g= P.augment_fn(gen_images)
    opt_D.zero_grad() 

    #do dammage logic
    D.module.set_prune_flag(True)
    aux = D(inputs_1, sg_linear=True, projection=True, projection2=False,clrOnly= True)
    view1_det = F.normalize(aux['projection']).detach()
    D.module.set_prune_flag(False)
    aux = D(inputs_2, sg_linear=True, projection=True, projection2=False,clrOnly= True)
    view2 = F.normalize(aux['projection'])
    simclr_loss = nt_xent(view1_det, view2 , temperature=P.temp)       
    simclr_loss.backward()
    D.module.set_prune_flag(True)
    aux = D(inputs_1, sg_linear=True, projection=True, projection2=False,clrOnly= True)
    view1 = F.normalize(aux['projection'])
    view2_det=view2.detach()
    simclr_loss = nt_xent(view1, view2_det , temperature=P.temp)       
    simclr_loss.backward()
    D.module.set_prune_flag(False)


    #now contraD contrast
    cat_images = torch.cat([inputs_1, inputs_2, inputs_g], dim=0)
    d_all,auxg = D(cat_images, sg_linear=True, projection=False, projection2=True)
    reals = auxg['projection2']
    reals = F.normalize(reals)
    real1, real2, fakes = reals[:N], reals[N:2*N], reals[2*N:]
    sup_loss = supcon_fake(real1, real2, fakes, temperature=P.temp, distributed=P.distributed)

    d_real, d_gen = d_all[:N], d_all[2*N:3*N]
    #finally calc the discriminator
    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()

    #update everything
    loss=sup_loss + d_loss # +  simclr_loss  #+simclr_loss1+simclr_loss2
    loss.backward()
    opt_D.step()

    return  sup_loss, {
        "penalty": d_loss,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }


def loss_G_fn(P, D, options, images, gen_images):
    # print ("damage gen")
    d_gen = D(P.augment_fn(gen_images))
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    elif options['loss'] == 'lsgan':
        g_loss = 0.5 * ((d_gen - 1.0) ** 2).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss


    # return simclr_loss1 + simclr_loss2 + P.lbd_a * sup_loss, {
    #     "penalty": d_loss,
    #     "d_real": d_real.mean(),
    #     "d_gen": d_gen.mean(),
    # }

    # elif options['loss'] == 'wgan':
    #     d_loss = d_gen.mean() - d_real.mean()
    # elif options['loss'] == 'hinge':
    #     d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    # elif options['loss'] == 'lsgan':
    #     d_loss_real = ((d_real - 1.0) ** 2).mean()
    #     d_loss_fake = (d_gen ** 2).mean()
    #     d_loss = 0.5 * (d_loss_real + d_loss_fake)
    # else:
    #     raise NotImplementedError()
# def loss_D_fn(P, D, options, images, gen_images):
#     # print ("damage discrim")
#     assert images.size(0) == gen_images.size(0)
#     gen_images = gen_images.detach()
#     N = images.size(0)

#     cat_images = torch.cat([images, images, gen_images], dim=0)
#     d_all, aux = D(P.augment_fn(cat_images), sg_linear=True, projection=True, projection2=True)
#     views = aux['projection']
#     views = F.normalize(views)
#     view1, view2, others = views[:N], views[N:2*N], views[2*N:]
#     simclr_loss = nt_xent(view1, view2, temperature=P.temp, distributed=P.distributed, normalize=False)

#     reals = aux['projection2']
#     reals = F.normalize(reals)
#     real1, real2, fakes = reals[:N], reals[N:2*N], reals[2*N:]
#     sup_loss = supcon_fake(real1, real2, fakes, temperature=P.temp, distributed=P.distributed)

#     d_real, d_gen = d_all[:N], d_all[2*N:3*N]
#     if options['loss'] == 'nonsat':
#         d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()
#     elif options['loss'] == 'wgan':
#         d_loss = d_gen.mean() - d_real.mean()
#     elif options['loss'] == 'hinge':
#         d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
#     elif options['loss'] == 'lsgan':
#         d_loss_real = ((d_real - 1.0) ** 2).mean()
#         d_loss_fake = (d_gen ** 2).mean()
#         d_loss = 0.5 * (d_loss_real + d_loss_fake)
#     else:
#         raise NotImplementedError()

#     return simclr_loss + P.lbd_a * sup_loss, {
#         "penalty": d_loss,
#         "d_real": d_real.mean(),
#         "d_gen": d_gen.mean(),
#     }
