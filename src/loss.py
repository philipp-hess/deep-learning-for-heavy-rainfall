import torch
import torch.nn as nn
import torch.nn.functional as F
from src.structural_similarity import MSSSIMLoss

class LossFunctions():

    def __init__(self,
                hparam,
                device='cpu'
                ):

        self.device = device

        self.alpha = hparam.alpha
        self.scale_pos = hparam.scale_pos
        self.scale_neg = hparam.scale_neg
        self.w_fact = torch.Tensor([hparam.w_fact]).to(device)
        self.w_exponent = torch.Tensor([hparam.w_exponent]).to(device)
        self.data_range = hparam.data_range

        self.zero = torch.Tensor([0]).to(self.device)
        self.one = torch.Tensor([1]).to(self.device)


    def mse(self, output, target):
        """ Mean Squared Error Loss """

        criterion = torch.nn.MSELoss()
        loss = criterion(output, target)
        return loss


    def msssim(self, output, target):
        """ Multi-Scale Structural Similarity Index Loss """

        criterion = MSSSIMLoss(data_range=self.data_range)
        loss = criterion(output, target)
        return loss


    def msssim_weighted_mse(self, output, target):
        """ MS-SSIM with Weighted Mean Squared Error Loss """

        weights = torch.minimum(self.one, self.w_fact*torch.exp(self.w_exponent*target))
        criterion = MSSSIMLoss(data_range=self.data_range)

        loss = self.alpha*(weights * (output - target) ** 2).mean() \
             + (1.-self.alpha)*criterion(output, target)

        return loss


    def mse_mae(self, output, target):
        """ Combined Mean Squared Error and Mean Absolute Error Loss """

        loss = (1.-self.alpha)*((output - target) ** 2).mean() \
               + self.alpha*(abs(output - target)).mean()
        return loss


    def weighted_mse(self, output, target):
        """ Weighted Mean Squared Error Loss """

        weights = torch.minimum(self.one, self.w_fact*torch.exp(self.w_exponent*target))

        loss = (weights * (output - target) ** 2).mean()
        return loss


    def mae_weighted_mse(self, output, target):
        """ Weighted Mean Squared Error and Mean Absolute Error Loss """

        weights = torch.minimum(self.one, self.w_fact*torch.exp(self.w_exponent*target))

        loss = self.alpha*(weights * (output - target) ** 2).mean() \
             + (1.-self.alpha)*(torch.abs(output - target)).mean()
        return loss
