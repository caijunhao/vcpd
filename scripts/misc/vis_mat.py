import matplotlib.pyplot as plt
import numpy as np

x_labels = ['{}'.format(i) for i in range(7)]
y_labels = [s for s in 'ABCDEFG'][::-1]
gsr = np.array([[0.614, 0.986, 0.506, 0.341, 0.294, 0.941, 0.525],
                [0.009, 0.977, 0.968, 0.998, 0.973, 0.876, 0.928],
                [0.998, 0.966, 0.776, 0.954, 0.958, 0.997, 0.993],
                [0.996, 1, 1, 0.973, 0.993, 0.944, 0.957],
                [1, 1, 0.965, 0.998, 0.973, 0.993, 0.989],
                [1, 0.972, 1, 0.957, 1, 0.999, 0.990],
                [1, 1, 0.999, 1, 0.994, 0.989, 1]])
print(gsr)
fig, ax = plt.subplots()
im = ax.imshow(1-gsr, cmap='Set2')
ax.set_xticks(np.arange(7), labels=x_labels)
ax.set_yticks(np.arange(7), labels=y_labels)

for i in range(7):
    for j in range(7):
        text = ax.text(j, i, '{:.1f}'.format(gsr[i, j]*100),
                       ha="center", va="center", color="w")
fig.tight_layout()
plt.show()
