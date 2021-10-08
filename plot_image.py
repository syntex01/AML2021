import data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

data = data.Data()
iid = ["b4fcb78a1832"] #["0f4bc6a7b8b5", "7e7d3afebf5d", "239d414106d4", "b1577bc61e24"]
#iid = data.images()[:10]
x, y, b = data.sample(iid, (2048, 2048), rgb=True, mode="scale")
print(b)

for i in range(len(iid)):
  print(iid[i], y[i])
  plt.title(f"{iid[i]}: {data.class_labels[np.argmax(y[i])]}")
  plt.imshow(x[i], aspect="equal")

  for box in b[i]:
    plt.gca().add_patch(Rectangle((box["x"], box["y"]), box["width"], box["height"], linewidth=1, edgecolor="r", facecolor="None"))

  # hide axis ticks
  plt.setp(plt.gca().get_xticklabels(), visible=False)
  plt.setp(plt.gca().get_yticklabels(), visible=False)
  #plt.savefig(f"archive/{iid[i]}_markings.png", format="png", dpi=300)
  plt.show()
