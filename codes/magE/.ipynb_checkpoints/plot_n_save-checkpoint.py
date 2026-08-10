import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

galname = sys.argv[1]
z = float(sys.argv[2])
x_liminf = float(sys.argv[3])
y_limsup = float(sys.argv[4])

data = pd.read_csv(galname + '.csv')

plt.figure(figsize=(15, 3))

plt.plot(data['wave']/(1 + z), data['flux'], color='black', label='Flux')
plt.plot(data['wave']/(1 + z), data['noise'], ls='--', color='purple',
        label='Noise')

plt.ylim(x_liminf, y_limsup)
plt.ylabel(r'Flux [ergs / cm$^{2} / s$]')
plt.xlabel(r'Rest-frame $\lambda$ [$\AA$]')
plt.legend(frameon=False)
plt.savefig(galname + '.png', bbox_inches='tight')
