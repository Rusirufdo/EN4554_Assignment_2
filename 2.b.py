import numpy as np
import matplotlib.pyplot as plt
import torch


def objective(xy):
    x, y = xy[0], xy[1]
    return 0.5 * (x**2 + 50 * y**2)


def gradient(xy):
    x, y = xy[0], xy[1]
    return torch.tensor([x, 50 * y], dtype=torch.float64)


def run_sgd(start, lr=0.03, steps=40):
    point = start.clone().detach().to(torch.float64)
    path = [point.clone()]
    for _ in range(steps):
        grad = gradient(point)
        point = point - lr * grad
        path.append(point.clone())
    return torch.stack(path)


def run_momentum(start, lr=0.03, momentum=0.9, steps=40):
    point = start.clone().detach().to(torch.float64)
    velocity = torch.zeros_like(point)
    path = [point.clone()]
    for _ in range(steps):
        grad = gradient(point)
        velocity = momentum * velocity + grad
        point = point - lr * velocity
        path.append(point.clone())
    return torch.stack(path)


def run_adam(start, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=40):
    point = start.clone().detach().to(torch.float64)
    m = torch.zeros_like(point)
    v = torch.zeros_like(point)
    path = [point.clone()]
    for step in range(1, steps + 1):
        grad = gradient(point)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        point = point - lr * (m_hat / (v_hat.sqrt() + eps))
        path.append(point.clone())
    return torch.stack(path)


if __name__ == '__main__':
    start_point = torch.tensor([8.0, 8.0], dtype=torch.float64)
    sgd_path = run_sgd(start_point, lr=0.03, steps=40)
    momentum_path = run_momentum(start_point, lr=0.03, momentum=0.9, steps=40)
    adam_path = run_adam(start_point, lr=0.1, steps=40)

    print('Final point with SGD:', sgd_path[-1])
    print('Final point with Momentum:', momentum_path[-1])
    print('Final point with Adam:', adam_path[-1])

    x_vals = np.linspace(-9, 9, 300)
    y_vals = np.linspace(-9, 9, 300)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 0.5 * (X**2 + 50 * Y**2)

    plt.figure(figsize=(9, 7))
    contours = plt.contour(X, Y, Z, levels=25)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.plot(sgd_path[:, 0].numpy(), sgd_path[:, 1].numpy(), marker='o', label='SGD')
    plt.plot(momentum_path[:, 0].numpy(), momentum_path[:, 1].numpy(), marker='s', label='SGD with Momentum')
    plt.plot(adam_path[:, 0].numpy(), adam_path[:, 1].numpy(), marker='^', label='Adam')
    plt.scatter(start_point[0].item(), start_point[1].item(), marker='x', s=100, label='Start')
    plt.scatter(0, 0, marker='*', s=150, label='Minimum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimizer Trajectories on $f(x,y)=\\frac{1}{2}(x^2+50y^2)$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
