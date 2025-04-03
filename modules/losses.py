import torch 
import torch.nn.functional as F 
import lpips

class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""
    def __init__(self, linear_calibration: bool = False, normalize: bool = False):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net='vgg', lpips=linear_calibration, pretrained=True) # Note: only 'vgg' valid as loss  
        self.normalize = normalize # If true, normalize [0, 1] to [-1, 1]
        
    def forward(self, predicted: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
        if predicted.ndim == 5: # 3D Image: Just use 2D model and compute average over slices 
            depth = predicted.shape[-1] 
            losses = torch.stack([
                self.loss_fn(predicted[:, 0, None, ..., d], 
                target[:, 0, None, ..., d], normalize=self.normalize) 
                for d in range(depth)
            ], dim=2)
            return torch.mean(losses, dim=2, keepdim=True)
        else:
            if predicted.shape[1] in [1, 3]:
                return self.loss_fn(predicted, target, normalize=self.normalize)
            else:
                losses = torch.stack([
                    self.loss_fn(predicted[:, i, None, ...], target[:, i, None, ...], normalize=self.normalize) 
                    for i in range(predicted.shape[1])
                ], dim=1)
                return torch.mean(losses, dim=1)
            

def exp_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.exp(-logits_real))
    loss_fake = torch.mean(torch.exp(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake)))
    return d_loss

def kl_gaussians(mean1, logvar1, mean2, logvar2):
    """ Compute the KL divergence between two gaussians."""
    return 0.5 * (logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2)-1.0)



 