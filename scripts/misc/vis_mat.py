import matplotlib.pyplot as plt
import numpy as np

x_labels = ['{}'.format(i) for i in range(7)]
y_labels = [s for s in 'ABCDEFG'][::-1]
gsr = np.array([[0, 0.98, 0.5, 0.4, 0, 0.975, 0.6],
                [0, 0.975, 0.965, 0.99, 0.97, 0.945, 0.95],
                [1, 0.96, 0.785, 0.955, 0.965, 1, 0.995],
                [1, 1, 0.995, 0.97, 0.985, 0.96, 0.95],
                [1, 1, 0.95, 1, 0.98, 1, 0.98],
                [1, 0.96, 1, 0.965, 0.995, 1, 0.99],
                [1, 1, 1, 1, 0.995, 0.98, 1]])
print(gsr)
fig, ax = plt.subplots()
im = ax.imshow(1-gsr, cmap='Set2')
ax.set_xticks(np.arange(7), labels=x_labels)
ax.set_yticks(np.arange(7), labels=y_labels)

for i in range(7):
    for j in range(7):
        text = ax.text(j, i, '{:.3f}'.format(gsr[i, j]),
                       ha="center", va="center", color="w")
fig.tight_layout()
plt.show()
