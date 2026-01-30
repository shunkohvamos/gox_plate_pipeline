import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("out")
out.mkdir(exist_ok=True)

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Example plot")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(out / "test.png", dpi=200)
print("saved:", out / "test.png")
