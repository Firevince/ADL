import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
import math


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    alphabars = torch.tensor(
        [math.cos((t/timesteps + s)/(1+s) * (math.pi/2))**2 for t in range(timesteps)])
    betas = torch.tensor([1 - (alphabars[t+1]/alphabars[t]) for t in range(timesteps-1)])
    betas = torch.cat((betas, torch.tensor([0.999])),0)
    return betas




def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    limit = 5

    for t in range(timesteps):
        x = -limit + (2*limit*t)/(timesteps)
        if t==0:
            betas = torch.tensor([beta_start + (beta_end - beta_start) * (1/(1+math.exp(-x)))])
        else:
            betas = torch.cat((betas, torch.tensor([beta_start + (beta_end - beta_start) * (1/(1+math.exp(-x)))])),0)
    return betas


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help


        # define alphas
        # TODO

        self.alphas = 1 - self.betas
        self.alphabars = torch.cumprod(self.alphas, dim=0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        self.qs = []

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.posteriors = []


    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, class_labels=None, p_uncond=0.1):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # Equation 11 in the paper
        # Equation (8) in ex02 writeup
        # Use our model (noise predictor) to predict the mean

        null_class_labels = torch.full_like(class_labels, 10)
        # print(class_labels, null_class_labels)  # Debug print

        predicted_noise = model.forward(x, t, class_labels=null_class_labels)
        if class_labels is not None:
            predicted_noise_conditional = model.forward(x, t, class_labels)
            predicted_noise = (1+p_uncond)*predicted_noise_conditional - p_uncond*predicted_noise

        z = torch.zeros_like(x) if t_index == 0 else torch.randn_like(x)

        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - abar_t) * epsilon) + sqrt(beta_t) * z
        x_t_minus_1 = (1 / torch.sqrt(self.alphas[t_index])) \
                * (x - predicted_noise * self.betas[t_index] / torch.sqrt(1-self.alphabars[t_index])) \
                + torch.sqrt(self.betas[t_index]) * z


        # TODO (2.2): The method should return the image at timestep t-1.

        return x_t_minus_1

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, class_labels=None, p_uncond=0.1):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        size = (batch_size, channels, image_size[0], image_size[1])



        x_t = torch.normal(mean=0,std=1,size=size, dtype=torch.float).to(self.device)
        for t in range(self.timesteps - 1, 0, -1):
            x_t = self.p_sample(model, x_t, torch.tensor([t]*batch_size).to(self.device), t, class_labels=class_labels, p_uncond=p_uncond)
            # x_t = self.p_sample(model,x_t,torch.tensor([t]*batch_size).to(self.device),t)
        
        # TODO (2.2): Return the generated images
        return x_t

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise== None:
            noise = torch.normal(mean=0, std=1, size=x_zero.shape, dtype=torch.float)

        
        # test = self.alphabars[t]
        # test2 = x_zero[test]

        x_t = (torch.sqrt(self.alphabars[t[0]])*x_zero[0] + torch.sqrt(1-self.alphabars[t[0]])*noise[0]).unsqueeze(0)
        # x_t = x_t.to(self.device)
        for i in range(1,len(t)):
            x_t = torch.vstack((x_t, (torch.sqrt(self.alphabars[t[i]])*x_zero[i] + torch.sqrt(1-self.alphabars[t[i]])*noise[i]).unsqueeze(0)))
        
        # x_t = torch.sqrt(self.alphabars[t])*x_zero + torch.sqrt(1-self.alphabars[t])*noise
        return x_t

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1", class_labels=None):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.

        if noise== None:
            noise = torch.normal(mean=0, std=1, size=x_zero.shape, dtype=torch.float).to(self.device)
        

        q_t = self.q_sample(x_zero, t, noise)

        predicted_noise = denoise_model.forward(q_t, t, class_labels)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, predicted_noise)

        else:
            raise NotImplementedError()

        return loss
