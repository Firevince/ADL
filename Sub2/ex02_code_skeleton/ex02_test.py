import unittest
import torch
import os
from numpy.testing import assert_almost_equal
import sys
from torchvision.utils import save_image
import matplotlib.pyplot as plt



sys.path.append(os.path.abspath(os.path.join("Sub2", 'ex02_code_skeleton')))

from ex02_helpers import load_image
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.test_values = torch.load("Sub2/ex02_code_skeleton/ex02_test_values.pt")
        self.scheduler = lambda x: linear_beta_schedule(0.001, 0.02, x)
        self.img_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_q_sample(self):
        local_values = self.test_values["q_sample"]
        diffusor = Diffusion(timesteps=local_values["timesteps"],
                             get_noise_schedule=self.scheduler, img_size=self.img_size)

        output = diffusor.q_sample(x_zero=local_values["x_zero"].to(self.device),
                                   t=local_values["t"].to(self.device), noise=local_values["noise"].to(self.device))
        assert_almost_equal(local_values["expected_output"].numpy(), output.to(torch.device("cpu")).numpy(), decimal=5)

    def test_q_sample_image(self):
        img = load_image("Sub2/test_image/bee.png")
        local_values = self.test_values["q_sample"]

        # scheduler = lambda x: linear_beta_schedule(0.001, 0.02, x)
        scheduler = lambda x: cosine_beta_schedule(x)
        diffusor = Diffusion(timesteps=local_values["timesteps"],
                             get_noise_schedule=scheduler, img_size=self.img_size)

        timestep = torch.tensor([50])

        output = diffusor.q_sample(x_zero=img.to(self.device),
                                         t=timestep.to(self.device))
        save_image(output, "Sub2/test_image/output.png")
        assert(True)

    def test_beta_schedulers(self):
        timesteps = torch.tensor(30)
        betas_linear = linear_beta_schedule(0.001, 1, timesteps)
        betas_cosine = cosine_beta_schedule(timesteps)
        betas_sigmoid = sigmoid_beta_schedule(0.001, 1, timesteps)
        plt.plot(betas_linear.numpy(), label="linear")
        plt.plot(betas_cosine.numpy(), label="cosine")
        plt.plot(betas_sigmoid.numpy(), label="sigmoid")
        plt.legend()
        plt.savefig("Sub2/test_image/beta_schedules.png")

        abars_linear = torch.cumprod(1 - betas_linear, dim=0)
        abars_cosine = torch.cumprod(1 - betas_cosine, dim=0)
        abars_sigmoid = torch.cumprod(1 - betas_sigmoid, dim=0)
        plt.figure()
        plt.plot(abars_linear.numpy(), label="linear")
        plt.plot(abars_cosine.numpy(), label="cosine")
        plt.plot(abars_sigmoid.numpy(), label="sigmoid")
        plt.legend()
        plt.savefig("Sub2/test_image/abar_schedules.png")

        assert(True)



def main():
    unittest.main()


if __name__ == "__main__":
    main()
