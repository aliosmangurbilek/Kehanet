# tests/test_suite.py
import numpy as np
import pytest

from core.tensor import Tensor
from losses.mse import mse_loss
from losses.cross_entropy import cross_entropy
from layers.dense import Dense
from optimizers.sgd import SGD

# Test Tensor operations

def test_tensor_add_and_matmul_and_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = a + b  # [4.0,6.0]
    print("Tensor addition result:", c.data)
    assert np.allclose(c.data, [4.0, 6.0])
    # matmul test
    m1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    m2 = Tensor([[2.0], [1.0]], requires_grad=True)
    m3 = m1 @ m2  # shape (2,1)
    print("Matrix multiplication result shape:", m3.data.shape)
    assert m3.data.shape == (2,1)

    # backward on c (vector): specify grad
    grad_vec = np.ones_like(c.data, dtype=np.float32)
    c.backward(grad_vec)
    print("Gradient of a after backward:", a.grad)
    print("Gradient of b after backward:", b.grad)
    assert np.allclose(a.grad, grad_vec)
    assert np.allclose(b.grad, grad_vec)

# Test MSE loss

def test_mse_loss_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([1.2, 1.9, 2.7])
    loss = mse_loss(x, y)
    print("MSE loss value:", loss.data)
    loss.backward()
    expected_grad = (2.0 / 3) * (x.data - y.data)
    print("Gradient of x after backward:", x.grad)
    assert np.allclose(x.grad, expected_grad, atol=1e-6)

# Test Cross-Entropy loss

def test_cross_entropy_backward():
    logits = Tensor(np.array([[2.0,1.0,0.1]], dtype=np.float32), requires_grad=True)
    labels = np.array([0])
    target = np.zeros((1,3), dtype=np.float32)
    target[np.arange(1), labels] = 1.0
    target = Tensor(target)

    loss = cross_entropy(logits, target)
    print("Cross-entropy loss value:", loss.data)
    loss.backward()

    # manual softmax
    exp_logits = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    expected_grad = (probs - target.data) / 1  # N=1
    print("Gradient of logits after backward:", logits.grad)
    assert np.allclose(logits.grad, expected_grad, atol=1e-6)

# Test Dense layer forward and backward

def test_dense_forward_backward():
    layer = Dense(2, 3, activation=None, bias=True)
    # set deterministic weights and bias
    layer.W.data[:] = 1.0
    layer.b.data[:] = 0.0

    x = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=False)
    out = layer(x)
    print("Dense layer output:", out.data)
    # expected output: [[1*1+2*1, 1*1+2*1, 1*1+2*1]] = [3,3,3]
    assert np.allclose(out.data, [[3.0, 3.0, 3.0]], atol=1e-6)

    # backward: propagate ones
    out.backward(np.ones_like(out.data, dtype=np.float32))
    # grad W should be x.T @ grad = [[1],[2]] @ [1,1,1] = [[1,1,1],[2,2,2]]
    print("Gradient of W after backward:", layer.W.grad)
    print("Gradient of b after backward:", layer.b.grad)
    assert np.allclose(layer.W.grad, np.array([[1,1,1],[2,2,2]], dtype=np.float32))
    assert np.allclose(layer.b.grad, np.array([[1,1,1]], dtype=np.float32))

# Test SGD optimizer

def test_sgd_step_and_zero_grad():
    param = Tensor(np.array([1.0,2.0], dtype=np.float32), requires_grad=True)
    param.grad = np.array([0.5,1.0], dtype=np.float32)
    optimizer = SGD([param], lr=0.1)
    optimizer.step()
    print("Parameter values after SGD step:", param.data)
    assert np.allclose(param.data, np.array([1.0-0.05,2.0-0.1], dtype=np.float32))
    optimizer.zero_grad()
    print("Gradient after zero_grad:", param.grad)
    assert param.grad is None
