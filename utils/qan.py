import torch as t


# Detach class and function
class Detach(t.autograd.Function):
    '''
    a function called detach that returns its input (unmodified) during the forward pass,
    and whose gradient during the backward pass is zero (thus detaching itself from the backward graph).
    '''

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return t.zeros_like(grad_y)


def detach(x):
    return Detach.apply(x)


# Two important class: GradScale & RoundPass
class GradScale(t.nn.Module):
    def forward(self, s, s_grad_scale):
        y = s
        y_grad = s / s_grad_scale
        return detach(y - y_grad) + y_grad


class RoundPass(t.nn.Module):
    def forward(self, x):
        y = x.round()
        y_grad = x
        return detach(y - y_grad) + y_grad


# Identity function
def identity(x, *args, **kwargs):
    return x


# Norm Identity function
def identity_wo_norm(x, *args, **kwargs):
    return x / 255


# Quantize class
class Quantize(t.nn.Module):
    '''Used for Activation & Weight'''

    def __init__(self, is_activation, bit):
        super(Quantize, self).__init__()
        self.s = t.nn.Parameter(t.ones(1))  # learnable parameter  # t.zeros(1)
        if is_activation:
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            # signed weight is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1

        self.grad_scale = GradScale()  #
        self.round_pass = RoundPass()  #

    def forward(self, x):
        # if self.s.data < 0:
        #     self.s.data = -1 * self.s.data
        s_grad_scale = (self.thd_pos * x.numel()) ** 0.5  # Get grad Factor
        s_scale = self.grad_scale(self.s, s_grad_scale)  # Get Scale (class is t.nn.Module, has forward())

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)  # class is t.nn.Module, has forward()
        x = x * s_scale
        return x


class QuanConv2d(t.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, quan_bit_w=None, quan_bit_a=None, bias=False, **kwargs):

        super(QuanConv2d, self).__init__(in_channels, out_channels, kernel_size, bias=bias, **kwargs)

        if (quan_bit_a is None) or (quan_bit_a is 0):
            self.quan_a = identity_wo_norm
        else:
            self.quan_a = Quantize(is_activation=True, bit=quan_bit_a)

        if (quan_bit_w is None) or (quan_bit_w is 0):
            self.quan_w = identity
        else:
            self.quan_w = Quantize(is_activation=False, bit=quan_bit_w)

        # if bias and (quan_bit_a is not None or quan_bit_w is not None):
        #     raise Exception('LSQ cannot quantize biases.')

    def forward(self, x):
        weight_quan = self.quan_w(self.weight)  # Weight pass
        act_quan = self.quan_a(x)  # Feature pass
        return self.conv2d_forward(act_quan, weight_quan)
