__all__ = ['train']

import os, sys, json, math, itertools
import pandas as pd, numpy as np
import warnings

from tqdm import tqdm
#from tqdm.notebook import tqdm

import torch

from .utils import sample, generate_steps
from .losses import MMD_loss, OT_loss1, Density_loss, Local_density_loss
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
from DeepRUOT.models import velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet, FNet, ODEFunc2, ODEFunc3
from DeepRUOT.utils import group_extract, sample, to_np, generate_steps, cal_mass_loss, cal_mass_loss_reduce, parser, _valid_criterions
import geomloss
from geomloss import SamplesLoss

import matplotlib.pyplot as plt 

def train_un1(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,

    lambda_density = 1.0,

    autoencoder=None, 
    use_emb=False,
    use_gae=False,

    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    device=None,
    logger=None,
    use_pinn=False,

    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=0.1,
    lambda_mass=1,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    best_model_path=None,
):

    # Create the indicies for the steps that should be used
    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value) # if not use_local_density else Local_density_loss()

    
    model.train()
    model.to(device)
    step=0
    print('begin local loss')
    # Initialize the minimum Otloss with a very high value
    min_ot_loss = float('inf')

    for batch in tqdm(range(n_batches), desc='Training batches', leave=True):
        # apply local loss
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (initial_size)).to(device)
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
            data_t0.to(device)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out]
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out:
                    continue
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                data_t1.to(device)
                time = torch.Tensor([t0, t1])
                time.to(device)
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)
                # prediction

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)
                # loss between prediction and sample t1

                relative_mass_now = relative_mass[i_mass]
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t=odeint(ODEFunc2(model),initial_state_energy,t,options=dict(step_size=0.01),method='euler')
                lnw_t_last = lnw_t[-1]
                mu = torch.exp(lnw_t_last)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                loss_ot = criterion(x_t[-1], data_t1, mu, nu)
                i_mass=i_mass+1
                local_mass_loss = cal_mass_loss(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0])
                lnw0=lnw_t_last.detach()
                data_t0=x_t[-1].detach()
            
                print('Otloss')
                print(loss_ot)
                print('mass loss')
                print(local_mass_loss)
                loss=(lambda_ot*loss_ot+lambda_mass*local_mass_loss)


                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    print('density loss')
                    print(density_loss)

                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
               # Detach the loss from the computation graph and get its scalar value
            current_ot_loss = loss_ot.item()
            
            # Check if the current Otloss is the new minimum
            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                # Save the model's state_dict
                torch.save(model.state_dict(), best_model_path)
                logger.info(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            batch_loss = batch_loss.to(device)
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses



def train_un1_reduce(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,

    lambda_density = 1.0,

    autoencoder=None, 
    use_emb=False,
    use_gae=False,

    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    device=None,
    logger=None,
    use_pinn=False,

    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=0.1,
    lambda_mass=1,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    best_model_path=None,
    pca_transform=None,
    use_mass = True,
    global_mass = False,
    mass_detach = True,
):

    # Create the indicies for the steps that should be used
    steps = generate_steps(groups)

    if reverse:
        groups_reverse = groups[::-1]
        steps_reverse = generate_steps(groups_reverse)
        relative_mass_reverse = torch.tensor(relative_mass).flip(dims=[0])
        relative_mass_reverse = relative_mass_reverse / relative_mass_reverse[0]

    
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value) # if not use_local_density else Local_density_loss()
    
    model.train()
    model.to(device)
    step=0
    print('begin local loss')
    # Initialize the minimum Otloss with a very high value
    min_ot_loss = float('inf')

    # Create progress bar with OT loss display
    pbar = tqdm(range(n_batches), desc='Pretraining')

    for i in pbar:
        loss_batch = 0
        # apply local loss
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
            data_t0.to(device)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] # TODO: Currently does not work if hold_out='random'. Do to_ignore before. 
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out: # TODO: This `if` can be deleted since the groups does not include the ho timepoint anymore
                    continue                              # i.e. it is always False. 
                optimizer.zero_grad()
                data_t0.to(device)
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                data_t1.to(device)
                time = torch.Tensor([t0, t1])
                time.to(device)
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)
                # prediction

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)
                # loss between prediction and sample t1

                relative_mass_now = relative_mass[i_mass] #/relative_mass[i_mass-1]
                m0 = torch.zeros_like(lnw0).to(device)
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0, m0)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t, m_t=odeint2(ODEFunc2(model, use_mass = use_mass),initial_state_energy, t, atol=1e-5, rtol=1e-5, method='euler', options=dict(step_size=0.1))
                lnw_t_last = lnw_t[-1]
                m_t_last = m_t[-1]
                if use_mass:
                    mu = torch.exp(lnw_t_last)
                else:
                    mu = torch.ones(data_t0.shape[0],1)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                if mass_detach:
                    loss_ot = criterion(x_t[-1], data_t1, mu, nu)
                else:
                    loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1)
                    mu = mu.to(device)
                    nu = nu.to(device)
                    x_t[-1] = x_t[-1].to(device)
                    data_t1 = data_t1.to(device)
                    loss_ot = loss_fn(mu, x_t[-1],nu, data_t1)
                i_mass=i_mass+1
                global_mass_loss = torch.norm(torch.sum(torch.exp(lnw_t_last)) - relative_mass_now, p=2)**2
                local_mass_loss = cal_mass_loss_reduce(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0],dim_reducer=pca_transform)
                if global_mass:
                    mass_loss = global_mass_loss + local_mass_loss
                else:
                    mass_loss = local_mass_loss
                lnw0=lnw_t_last.detach()
                data_t0=x_t[-1].detach()
            
                logger.info(f'Otloss {loss_ot.item()}')
                logger.info(f'mass loss {mass_loss.item()}')
                logger.info(f'energy loss {m_t_last.mean().item()}')
                loss=(lambda_ot*loss_ot+lambda_mass*mass_loss + lambda_energy * m_t_last.mean())
                

                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    logger.info('density loss')
                    logger.info(density_loss)

                loss_batch += loss
                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
               # Detach the loss from the computation graph and get its scalar value
            current_ot_loss = loss_ot.item()
            
            # Update progress bar with current OT loss
            pbar.set_postfix({'OT Loss': f'{current_ot_loss:.6f}'})
            
            # Check if the current Otloss is the new minimum
            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                # Save the model's state_dict
                torch.save(model.state_dict(), best_model_path)
                logger.info(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        

            if reverse:
                i_mass_reverse=1
                lnw0_reverse = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)
                data_t0_reverse = sample(df, steps_reverse[0][0], size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                data_t0_reverse.to(device)

                batch_loss = []
                if hold_one_out:
                    groups = [g for g in groups if g != hold_out]
                    steps = generate_steps(groups)
                for step_idx, (t0, t1) in enumerate(steps_reverse):  
                    if hold_out in [t0, t1] and hold_one_out:
                        continue                              
                    optimizer.zero_grad()
                    data_t0_reverse.to(device)
                    size1=(df[df['samples']==t1].values.shape[0],)
                    data_t1_reverse = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                    data_t1_reverse.to(device)
                    time = torch.Tensor([t0, t1])
                    time.to(device)
                    if add_noise:
                        data_t0_reverse += noise(data_t0_reverse) * noise_scale
                        data_t1_reverse += noise(data_t1_reverse) * noise_scale
                    if autoencoder is not None and use_gae:
                        data_t0_reverse = autoencoder.encoder(data_t0_reverse)
                        data_t1_reverse = autoencoder.encoder(data_t1_reverse)

                    if autoencoder is not None and use_emb:        
                        data_tp_reverse, data_t1_reverse = autoencoder.encoder(data_tp_reverse), autoencoder.encoder(data_t1_reverse)

                    relative_mass_now = relative_mass_reverse[i_mass_reverse]
                    m0 = torch.zeros_like(lnw0).to(device)
                    data_t0_reverse=data_t0_reverse.to(device)
                    data_t1_reverse=data_t1_reverse.to(device)
                    initial_state_energy = (data_t0_reverse, lnw0_reverse, m0)
                    t=time.to(device)
                    t.requires_grad=True
                    data_t0_reverse.requires_grad=True
                    lnw0_reverse.requires_grad=True
                    
                    x_t_reverse, lnw_t_reverse, m_t_reverse=odeint(ODEFunc2(model, use_mass = use_mass),initial_state_energy, t, atol=1e-5, rtol=1e-5, method='euler', options=dict(step_size=0.1))
                    lnw_t_last_reverse = lnw_t_reverse[-1]
                    m_t_last_reverse = m_t_reverse[-1]
                    if use_mass:
                        mu_reverse = torch.exp(lnw_t_last_reverse)
                    else:
                        mu_reverse = torch.ones(data_t0_reverse.shape[0],1)
                    mu_reverse = mu_reverse / mu_reverse.sum()
                    nu_reverse = torch.ones(data_t1_reverse.shape[0],1)
                    nu_reverse = nu_reverse / nu_reverse.sum()
                    mu_reverse = mu_reverse.squeeze(1)
                    nu_reverse=nu_reverse.squeeze(1)
                    loss_ot_reverse = criterion(x_t_reverse[-1], data_t1_reverse, mu_reverse, nu_reverse)
                    i_mass_reverse=i_mass_reverse+1
                    global_mass_loss = torch.abs(torch.sum(torch.exp(lnw_t_last_reverse)) - relative_mass_now)/relative_mass_now
                    local_mass_loss = cal_mass_loss_reduce(data_t1_reverse, x_t_reverse[-1], lnw_t_last_reverse , relative_mass_now, sample_size[0],dim_reducer=pca_transform)
                    if global_mass:
                        mass_loss = global_mass_loss + local_mass_loss
                    else:
                        mass_loss = local_mass_loss
                    lnw0_reverse=lnw_t_last_reverse.detach()
                    data_t0_reverse=x_t_reverse[-1].detach()
                
                    logger.info(f'Otloss: {loss_ot_reverse}')
                    logger.info(f'mass loss: {mass_loss}')
                    logger.info(f'energy loss: {m_t_last_reverse.mean()}')
                    loss_reverse=(lambda_ot*loss_ot_reverse+lambda_mass*mass_loss - lambda_energy * m_t_last_reverse.mean())

                    if use_density_loss:                
                        density_loss = density_fn(data_t0_reverse, data_t1_reverse, top_k=top_k)
                        density_loss = density_loss.to(loss_reverse.device)
                        loss_reverse += lambda_density * density_loss
                        logger.info('density loss')
                        logger.info(density_loss)

                    loss_batch += loss_reverse
                    if apply_losses_in_time and local_loss:
                        loss_reverse.backward()
                        optimizer.step()
                        model.norm=[]
            batch_loss = torch.Tensor(batch_loss).float()
            batch_loss = batch_loss.to(device)
            
            if not apply_losses_in_time:
                loss_batch.backward()
                optimizer.step()

            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())




                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses


from .utils import density1, trace_df_dz
import torch, torch.nn as nn

def train_all(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,

    lambda_density = 1.0,
    datatime0=None,
    device=None,
    autoencoder=None, 
    use_emb=False,
    use_gae=False,

    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    
    logger=None,
    use_pinn=False,
    sf2m_score_model=None,
    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=10,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    sigmaa=0.1,
    lambda_initial=None,
    lambda_mass = 1,
    use_mass = True,
    exp_dir = None,
    scheduler = None,
    global_mass = True,
    pca_transform=None,
):

    # Disable embedding if autoencoder is None
    if autoencoder is None and (use_emb or use_gae):
        use_emb = False
        use_gae = False
        warnings.warn('\'autoencoder\' is \'None\', but \'use_emb\' or \'use_gae\' is True, both will be set to False.')

    # Set up noise function based on distribution type
    noise_fn = torch.randn if use_gaussian else torch.rand
    def noise(data):
        return noise_fn(*data.shape).to(device)
        
    # Generate time steps for forward pass
    steps = generate_steps(groups)

    # Generate time steps and masses for reverse pass if needed
    if reverse:
        groups_reverse = groups[::-1]
        steps_reverse = generate_steps(groups_reverse)
        relative_mass_reverse = torch.tensor(relative_mass).flip(dims=[0])
        relative_mass_reverse = relative_mass_reverse / relative_mass_reverse[0]

    # Initialize loss storage
    batch_losses = []
    globe_losses = []
    
    # Set up local loss tracking based on hold-out configuration
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value)
    
    model.train()
    step=0
    logger.info('begin local loss')

    # Track best OT loss for model checkpointing
    min_ot_loss = float('inf')

    pbar = tqdm(range(n_batches), desc='Training')
    for batch in pbar:

        # Local loss computation
        if local_loss and not global_loss:
            # Sample initial data points
            size0=(df[df['samples']==0].values.shape[0],)
            data_0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
            P0=0
            num=data_0.shape[0]
            time=torch.tensor([0.0]).to(device)
            time=time.expand(num,1)
            data_0=data_0.to(device)
            sf2m_score_model=sf2m_score_model.to(device)
            
            # Calculate score function values
            s2=sf2m_score_model(time, data_0) # log density
            data_0.requires_grad_(True)

            # Initialize weights and mass
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)  # Log normalized weights
            m0 = (torch.zeros(sample_size[0],1) / (initial_size)).to(device)  
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out]
                steps = generate_steps(groups)
                
            # Iterate through time steps
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out:
                    continue                              
                optimizer.zero_grad()

                # Sample target data
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                time = torch.Tensor([t0, t1])
                time.to(device)

                # Add noise if specified
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                    
                # Apply autoencoder if using GAE
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)

                # Apply autoencoder for embeddings
                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)

                # Get current relative mass target
                relative_mass_now = relative_mass[i_mass]
                
                # Move data to device
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                lnw0=lnw0.to(device)
                m0=m0.to(device)
                model=model.to(device)
                
                # Set up initial state
                initial_state_energy = (data_t0, lnw0,m0)
                t=time.to(device)
                t.requires_grad=True
                
                # Enable gradients
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                m0.requires_grad=True
                
                # Solve ODE
                x_t, lnw_t,m_t=odeint2(ODEFunc3(model,sf2m_score_model,sigmaa, use_mass),initial_state_energy,t,options=dict(step_size=0.1),atol=1e-5, rtol=1e-5, method='euler')
                lnw_t_last = lnw_t[-1]
                m_t_last=m_t[-1]
                
                # Calculate weights
                if use_mass:
                    mu = torch.exp(lnw_t_last)
                else:
                    mu = torch.ones(data_t0.shape[0],1, device = data_t0.device)
                    
                    
                # Normalize weights
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1, device = data_t0.device)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                
                # Calculate OT loss
                loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1)
                loss_ot = loss_fn(mu, x_t[-1],nu, data_t1,)
                i_mass=i_mass+1
                
                # Calculate mass losses
                global_mass_loss = torch.norm(torch.sum(torch.exp(lnw_t_last)) - relative_mass_now, p=2)**2
                local_mass_loss = cal_mass_loss_reduce(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0], dim_reducer=pca_transform)
                if global_mass:
                    mass_loss = global_mass_loss + local_mass_loss
                else:
                    mass_loss = local_mass_loss
                    
                # Update state for next iteration
                m0=m_t_last.clone().detach()
                lnw0=lnw_t_last.clone().detach()
                data_t0=x_t[-1].clone().detach()
            
                # Print loss components
                logger.info(f'Otloss {loss_ot.item()}')
                logger.info(f'mass loss {mass_loss.item()}')
                logger.info(f'energy loss {m_t_last.mean().item()}')
                loss_ot=loss_ot.to(device)
                # Combine losses
                loss=(lambda_ot*loss_ot+lambda_mass*mass_loss+m_t_last.mean()* lambda_energy)
                logger.info(f"total loss {loss}")

                # Add density loss if specified
                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    logger.info('density loss')
                    logger.info(density_loss)

                # Add PINN loss if specified
                if use_pinn:
                    P1=0
                    nnum=data_t1.shape[0]
                    ttime=time[1]
                    ttime = ttime.to(device)
                    vv, gg, _, _ = model(ttime, data_t1)
                    ttime=ttime.expand(nnum,1)
                    ss=sf2m_score_model(ttime, data_t1)
                    rrho = torch.exp(ss*2/sigmaa**2) 
                    rrho_t = torch.autograd.grad(outputs=rrho, inputs=ttime, grad_outputs=torch.ones_like(rrho),create_graph=True)[0]
                    vv_rho = vv * rrho
                    ddiv_v_rho = trace_df_dz(vv_rho, data_t1).unsqueeze(1)
                    if use_mass:
                        ppinn_loss = torch.abs(rrho_t + ddiv_v_rho - gg * rrho)
                    else:
                        ppinn_loss = torch.abs(rrho_t + ddiv_v_rho)
                    pppinn_loss = ppinn_loss

                    mean_pppinn_loss = torch.mean(pppinn_loss)

                    # Calculate initial density loss
                    size0=(df[df['samples']==0].values.shape[0],)
                    data_0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                    P0=0
                    num=data_0.shape[0]
                    time=torch.tensor([0.0]).to(device)
                    time=time.expand(num,1)
                    data_0=data_0.to(device)
                    sf2m_score_model=sf2m_score_model.to(device)
                    s2=sf2m_score_model(time, data_0)
                    data_0.requires_grad_(True)
                    density_values = density1(data_0,datatime0,device)
                    loss2=0
                    loss2=torch.mean((torch.exp(s2*2/sigmaa**2)-density_values)**2)
  
                    logger.info('pinloss')
                    logger.info(mean_pppinn_loss+loss2)
                    
                    loss += lambda_pinn * mean_pppinn_loss+lambda_initial*loss2

                # Apply local loss if specified
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    if scheduler != None:
                        scheduler.step()
                    model.norm=[]
                batch_loss.append(loss)
            
            # Check for new best model
            current_ot_loss = loss_ot.item()
            if current_ot_loss < min_ot_loss and exp_dir != None:
                min_ot_loss = current_ot_loss
                if use_pinn:
                    torch.save(sf2m_score_model.state_dict(), os.path.join(exp_dir, 'score_model_result'))
                torch.save(model.state_dict(), os.path.join(exp_dir, 'model_result'))
                logger.info(f'New minimum otloss found: {min_ot_loss}. Model saved.')
            pbar.set_postfix({'OT Loss': f'{current_ot_loss:.6f}'})
            # Reverse pass if specified
            if reverse:
                i_mass_reverse=1
                lnw0_reverse = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)
                data_t0_reverse = sample(df, steps_reverse[0][0], size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                data_t0_reverse.to(device)

                batch_loss = []
                if hold_one_out:
                    groups = [g for g in groups if g != hold_out]
                    steps = generate_steps(groups)
                    
                # Iterate through reverse time steps
                for step_idx, (t0, t1) in enumerate(steps_reverse):  
                    if hold_out in [t0, t1] and hold_one_out:
                        continue                              
                    optimizer.zero_grad()
                    
                    data_t0_reverse.to(device)
                    size1=(df[df['samples']==t1].values.shape[0],)
                    data_t1_reverse = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, device = device)
                    data_t1_reverse.to(device)
                    time = torch.Tensor([t0, t1])
                    time.to(device)
                    
                    # Add noise if specified
                    if add_noise:
                        data_t0_reverse += noise(data_t0_reverse) * noise_scale
                        data_t1_reverse += noise(data_t1_reverse) * noise_scale
                        
                    # Apply autoencoder if using GAE
                    if autoencoder is not None and use_gae:
                        data_t0_reverse = autoencoder.encoder(data_t0_reverse)
                        data_t1_reverse = autoencoder.encoder(data_t1_reverse)

                    # Apply autoencoder for embeddings
                    if autoencoder is not None and use_emb:        
                        data_tp_reverse, data_t1_reverse = autoencoder.encoder(data_tp_reverse), autoencoder.encoder(data_t1_reverse)

                    # Get current relative mass target
                    relative_mass_now = relative_mass_reverse[i_mass_reverse]
                    m0 = torch.zeros_like(lnw0).to(device)
                    data_t0_reverse=data_t0_reverse.to(device)
                    data_t1_reverse=data_t1_reverse.to(device)
                    initial_state_energy = (data_t0_reverse, lnw0_reverse, m0)
                    t=time.to(device)
                    t.requires_grad=True
                    data_t0_reverse.requires_grad=True
                    lnw0_reverse.requires_grad=True
                    
                    # Solve reverse ODE
                    x_t_reverse, lnw_t_reverse, m_t_reverse=odeint2(ODEFunc3(model,sf2m_score_model,sigmaa, use_mass),initial_state_energy, t, atol=1e-5, rtol=1e-5, method='euler', options=dict(step_size=0.1))
                    lnw_t_last_reverse = lnw_t_reverse[-1]
                    m_t_last_reverse = m_t_reverse[-1]
                    
                    # Calculate weights
                    if use_mass:
                        mu_reverse = torch.exp(lnw_t_last_reverse)
                    else:
                        mu_reverse = torch.ones(data_t0_reverse.shape[0],1)
                    mu_reverse = mu_reverse / mu_reverse.sum()
                    nu_reverse = torch.ones(data_t1_reverse.shape[0],1)
                    nu_reverse = nu_reverse / nu_reverse.sum()
                    mu_reverse = mu_reverse.squeeze(1)
                    nu_reverse=nu_reverse.squeeze(1)
                    
                    # Calculate reverse OT loss
                    loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1)
                    loss_ot_reverse = loss_fn(mu_reverse, x_t_reverse[-1],nu_reverse, data_t1_reverse,)
                    i_mass_reverse=i_mass_reverse+1
                    
                    # Calculate reverse mass losses
                    global_mass_loss = torch.norm(torch.sum(torch.exp(lnw_t_last_reverse)) - relative_mass_now, p=2)**2
                    local_mass_loss = cal_mass_loss_reduce(data_t1_reverse, x_t_reverse[-1], lnw_t_last_reverse , relative_mass_now, sample_size[0],dim_reducer=pca_transform)
                    if global_mass:
                        mass_loss = global_mass_loss + local_mass_loss
                    else:
                        mass_loss = local_mass_loss
                        
                    # Update state for next iteration
                    lnw0_reverse=lnw_t_last_reverse.detach()
                    data_t0_reverse=x_t_reverse[-1].detach()
                
                    # Print reverse loss components
                    logger.info(f'Otloss {loss_ot_reverse.item()}')
                    logger.info(f'mass loss {mass_loss.item()}')
                    logger.info(f'energy loss {m_t_last_reverse.mean().item()}')
                    
                    # Combine reverse losses
                    loss_reverse=(lambda_ot*loss_ot_reverse+lambda_mass*mass_loss - lambda_energy * m_t_last_reverse.mean())

                    # Add density loss if specified
                    if use_density_loss:                
                        density_loss = density_fn(data_t0_reverse, data_t1_reverse, top_k=top_k)
                        density_loss = density_loss.to(loss_reverse.device)
                        loss_reverse += lambda_density * density_loss
                        logger.info('density loss')
                        logger.info(density_loss)

                    # Apply reverse local loss if specified
                    if apply_losses_in_time and local_loss:
                        loss_reverse.backward()
                        optimizer.step()
                        model.norm=[]
        
            # Convert batch losses to tensor
            batch_loss = torch.Tensor(batch_loss).float()
            batch_loss = batch_loss.to(device)
            
            # Apply accumulated loss if not applying losses in time
            if not apply_losses_in_time:
                torch.mean(batch_loss).backward()
                optimizer.step()
                if scheduler != None:
                    scheduler.step()

            # Store batch statistics
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
            
        
                     
    # Print final training loss
    print_loss = globe_losses if global_loss else batch_losses 
    logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses

