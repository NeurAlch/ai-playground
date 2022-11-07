# binary classifier

from neuron import MLP

x = [2., 3., -1.]
n = MLP(3, [4, 4, 1])

dataset = [
    [2., 3., -1],   # 1
    [3., -1., .5],  # -1
    [.5, 1., 1.],   # -1
    [1., 1., -1.],  # 1
]

ys = [1., -1., -1., 1.]  # labels / targets

epochs = 50
step_size = 0.05
ypred = []

for e in range(epochs):
    # forward pass
    ypred = [n(x) for x in dataset]
    # ygt = y ground truth
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    # backward pass
    # zero grad
    # https://twitter.com/karpathy/status/1013244313327681536
    for p in n.parameters():
        p.zero_grad()
    # minize the loss
    loss.backward()

    # update, stochastic gradient descent
    for p in n.parameters():
        p.data += -step_size * p.grad

    print(f'Step {e + 1}; Loss is {loss.data}')

print([yp.data for yp in ypred])
