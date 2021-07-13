## Running all the required functions
from pathlib import Path; home = str(Path.home())
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import GlabTop2

## Vergleich van der Tricht und Farinotti

farinotti = pd.read_csv("/home/ana/Desktop/Test_thickness/Farinotti.csv")
tricht = pd.read_csv("/home/ana/Desktop/Test_thickness/vander_tricht.csv")
glabtop = pd.read_csv("/home/ana/Desktop/Test_thickness/glabtop.csv")

farinotti = farinotti.sort_values('elev_min')
tricht = tricht.sort_values('elev_min')
glabtop = glabtop.sort_values('elev_min')


plt.plot(farinotti["elev_min"], farinotti["_mean"], label="Farinotti")
plt.plot(tricht["elev_min"], tricht["_mean"], label="van der Tricht")
plt.plot(glabtop["elev_min"], glabtop["_mean"], label="Glabtop")
plt.legend()
plt.xlabel("Elevation [m]"), plt.ylabel("Thickness [m]")
plt.show()