import torch

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.pre_w = None
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):

        if self.pre_w is None:
            # print('no pre_w')
            self._init_param()

        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = self.module.weight

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        # print('sigma1: '+str(sigma))
        # print('sigma1: '+str(u.dot(self.pre_w.view(height, -1).mv(v))))
        sigma_amp = torch.abs(u.dot(self.pre_w.view(height, -1).mv(v))/sigma).data
        if sigma_amp > 1:
            sigma_amp = 1/sigma_amp
        # print('sigma_amp: '+str(sigma_amp))
        sigma = torch.pow(sigma, sigma_amp)
        # print('sigma2: '+str(sigma))

        w.data = w / sigma.expand_as(w)
        self.pre_w = w.clone()

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = self.module.weight

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)

    def _init_param(self):
        w = self.module.weight
        u = self.module.weight_u
        v = self.module.weight_v

        height = w.data.shape[0]
        u_mat, s, v_mat = torch.svd(w.view(height, -1).data)
        # print(u_mat.data.size())
        # print(v_mat.data.size())
        # print(s.data.size())
        u.data = u_mat[:, 0]
        v.data = v_mat[:, 0]
        w.data = w.data / s[0].data
        self.pre_w = w.data

    def forward(self, *args):
        if torch.is_grad_enabled() and self.module.training:
            # print('xxxxx')
            self._update_u_v()
        return self.module.forward(*args)



def main():

    x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).view(1, 2, 3, 3),requires_grad=True)
    conv = nn.Conv2d(2, 2, 3, stride=1, padding=(1,1))
    conv2 = nn.Conv2d(2, 2, 3, stride=1, padding=(1,1))
    addition = SpectralNorm(conv2)

    a = addition(x)
    a = torch.nn.functional.sigmoid(a)
    z = conv(a)
    # print(x)
    # print(z.data)
    # print(y.data)
    print(z.data)
    print(conv.weight)
    print(conv2.weight)
    # print(x.grad)
    opt = torch.optim.Adam(list(conv.parameters())+list(addition.parameters()), lr=0.1)
    for i in range(40):
        opt.zero_grad()
        a = addition(x)
        a = torch.nn.functional.sigmoid(a)
        z = conv(a)
        y = nn.functional.mse_loss(z, x.data)
        y.backward()
        opt.step()

    print(z.data)
    print(conv.weight)
    print(conv2.weight)

    a = addition(x)
    a = torch.nn.functional.sigmoid(a)
    z = conv(a)

    print('-----')
    print(z)


if __name__ == '__main__':
    main()