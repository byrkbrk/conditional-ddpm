import torch
import numpy as np

# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, a_t, b_t, ab_t, z=None):
    if z is None:
        z = torch.randn_like(x)
    if t == 1: # based on algorithm 2
        z = 0
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, n_channel, height, width, nn_model, timesteps, a_t, b_t, ab_t, 
                device, context=None, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, n_channel, height, width).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, a_t, b_t, ab_t, z)
        if i % save_rate ==0 or i==timesteps or i<8:
            #intermediate.append(samples.detach().cpu().numpy())
            intermediate.append(samples.detach().cpu())
    #intermediate = np.stack(intermediate)
    return samples, intermediate


# incorrectly sample without adding in noise
@torch.no_grad()
def sample_ddpm_incorrect(n_sample, height, width, nn_model, timesteps, a_t, b_t, ab_t, 
                device, context=None, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, width).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = 0

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, a_t, b_t, ab_t, z)
        if i % save_rate ==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


# define sampling function for DDIM   
# removes the noise using ddim
def denoise_ddim(x, t, t_prev, pred_noise, ab_t):
    ab = ab_t[t]
    ab_prev = ab_t[t_prev]
    
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    return x0_pred + dir_xt


# fast sampling algorithm with context
@torch.no_grad()
def sample_ddim(n_sample, height, width, nn_model, timesteps, ab_t, 
                device, context=None, n=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, width).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = timesteps // n
    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps, ab_t)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate