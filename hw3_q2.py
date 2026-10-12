import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().

    Routines provided as hints: torch.outer, torch.clamp, torch.Tensor.backward,
    torch.tensor.detach, with torch.no grad():, torch.Tensor.requires grad ,
    torch.tensor.grad.zero ,
    '''
    
    num_samples = x_train.shape[0]
    kernel_matrix = torch.zeros(num_samples, num_samples, dtype=x_train.dtype)
    
    for i in range(num_samples):
        for j in range(num_samples):
            kernel_matrix[i, j] = kernel(x_train[i], x_train[j])

    label_kernel_matrix = (torch.outer(y_train, y_train) * kernel_matrix).detach()
    alpha = torch.zeros(num_samples, dtype=label_kernel_matrix.dtype, requires_grad=True)

    for i in range(num_iters):
        loss = 0.5 * (alpha @ label_kernel_matrix @ alpha) - alpha.sum()
        loss.backward()

        with torch.no_grad():
            alpha -= lr * alpha.grad
            if c is None:
                alpha.clamp_(min=0)
            else:
                alpha.clamp_(min=0, max=c)
            alpha.grad.zero_()

    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    kernel_matrix = torch.zeros(num_test, num_train, dtype=x_train.dtype)

    for i in range(num_test):
        for j in range(num_train):
            kernel_matrix[i, j] = kernel(x_test[i], x_train[j])

    return kernel_matrix @ (alpha * y_train)
