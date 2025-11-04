import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
species = "HCNOS"
filename = f"derivative-test-{species}-FCC-ALLEGRO-10-600.csv"
df = pd.read_csv(filename)

# filter by "good" energy values
min_e = 0.1
max_e = 1000
df = df[abs(df['energy']) < max_e]
df = df[abs(df['energy']) > min_e]
df = df.sort_values(by='alat')

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot energy with markers
color = 'tab:blue'
ax1.set_xlabel('alat (Å)', fontsize=14)
ax1.set_ylabel('Shifted Energy (log scale)', color=color, fontsize=14)
ax1.set_yscale('log')
ax1.plot(df['alat'], df['energy'] + abs(df['energy'].min()), 
         color=color, linestyle='-',linewidth=4,
         label='Shifted Energy'
         )
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)

# Create second y-axis for errmax
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('errmax (log scale)', color=color, fontsize=14)
ax2.set_yscale('log')
ax2.plot(df['alat'], df['errmax'], 
         color=color, linestyle=':', linewidth=4,
         label='errmax')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

# Title and grid
plt.title(f'Log Plot of Shifted-Energy and errmax vs alat for {species} FCC\nFiltered for {min_e}eV < |energy| < {max_e}eV', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=12)

# Set font size for axis ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax1.tick_params(axis='x', size=8)  # Increase tick size (length) for x-axis

# Layout and display
fig.tight_layout()
plt.show()
