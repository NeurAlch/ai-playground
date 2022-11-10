import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples_per_class = 1000

# shape (1000, 2)
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1., .5], [.5, 1.]],  # 2D points
    size=num_samples_per_class,
)

# shape (1000, 2)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[.5, 1.], [1., .5]],  # 2D points
    size=num_samples_per_class,
)

# shape (2000, 2)
inputs = np.vstack((
    negative_samples,
    positive_samples,
)).astype(np.float32)

# target labels (0, 1)
targets = np.vstack((
    np.zeros((num_samples_per_class, 1), dtype=np.float32),  # negative samples
    np.ones((num_samples_per_class, 1), dtype=np.float32),  # positive samples
))

targets_2 = np.vstack((
    np.ones((num_samples_per_class, 1), dtype=np.float32),  # positive samples
    np.zeros((num_samples_per_class, 1), dtype=np.float32),  # negative samples
))

# 2D points
input_dim = 2
# single score per sample (close to 0 is class 0, close to 1 is class 1)
output_dim = 1

# initialize our weights
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))

# initialize our biases
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# forward pass
def model(inputs):
    return tf.matmul(inputs, W) + b


# loss function
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    # average loss scores to a single scalar
    return tf.reduce_mean(per_sample_losses)


# larger learning rate since we are using batch training and not mini-batch training
learning_rate = 0.1


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)

    # compute gradients
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])

    # update weights
    W.assign_sub(learning_rate * grad_loss_wrt_W)
    b.assign_sub(learning_rate * grad_loss_wrt_b)

    return loss


# training loop
for step in range(50):
    loss = training_step(inputs, targets)
    print(f"1: Epoch: {step}, Loss: {loss}")

predictions = model(inputs)

# re-train for new targets
for step in range(25):
    loss = training_step(inputs, targets_2)
    print(f"2: Epoch: {step}, Loss: {loss}")

predictions_2 = model(inputs)

x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, '-r')
plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=predictions[:, 0] > 0.5
)
plt.savefig('linear-classification-tensorflow-multiple-1.png')

plt.clf()
plt.plot(x, y, '-r')
plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=predictions_2[:, 0] > 0.5
)
plt.savefig('linear-classification-tensorflow-multiple-2.png')
