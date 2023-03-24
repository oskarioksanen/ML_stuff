import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.special import expit

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Create five points to describe the height of hobbits and elves
np.random.seed(13)
x_hobbit = np.random.normal(1.1, 0.3, 5)
x_elf = np.random.normal(1.9, 0.4, 5)

# Let's assign each x points to suitable output, hobbit 0 and elf 1
y_hobbit = np.zeros(x_hobbit.shape)
y_hobbit[:] = 0.0
y_elf = np.zeros(x_elf.shape)
y_elf[:] = 1.0

# Let's combine training data to one vector
x_tr = np.concatenate((x_hobbit, x_elf))
y_tr = np.concatenate((y_hobbit, y_elf))

# Initial weights initialized at zero
w0_t = 0
w1_t = 0

# Let's calculate the MSE with initial weight values (guesses)
y_pred_own = sigmoid(w1_t * x_tr + w0_t)
y_pred_expit = expit(w1_t * x_tr + w0_t)

MSE_own = np.sum((y_tr-y_pred_own)**2)/len(y_tr)
MSE_expit = np.sum((y_tr-y_pred_expit)**2)/len(y_tr)

print(MSE_own)
print(MSE_expit)

num_of_epochs = 100
learning_rate = 0.5

for e in range(num_of_epochs):
    for x_ind, x in enumerate(x_tr):
        y = sigmoid(w1_t * x + w0_t)
        w1_t = w1_t+learning_rate*(y_tr[x_ind]-y)*x
        w0_t = w0_t+learning_rate*(y_tr[x_ind]-y)*1

    if np.mod(e, 10) == 0 or e == 1:
        y_pred = sigmoid(w1_t * x + w0_t)
        MSE = np.sum((y_tr - y_pred)**2)/(len(y_tr))
        plt.title(f'Epoch={e} w0={w0_t:.2f} w1={w1_t:.2f} MSE={MSE:.2f}')
        plt.plot(x_hobbit, y_hobbit, 'co', label="hobbit")
        plt.plot(x_elf, y_elf, 'mo', label="elf")
        x = np.linspace(0.0, 5.0, 50)
        plt.plot(x, sigmoid(w1_t * x + w0_t), 'b-', label="y=logsig(w1*x+w0")
        plt.plot([0.5, 5.0], [0.5, 5.0], 'k--', label="y=0 (class boundary)")
        plt.xlabel("height [m]")
        plt.legend()
        plt.show()
