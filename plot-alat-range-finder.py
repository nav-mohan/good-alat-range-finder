import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file

#species, model = "CdTe", "SW_WangStroudMarkworth"
#species,model = "Si","SW_LeeHwang"
species,model = "Al", "EAMCubinNaturalSpline"
#species,model="Si", "StillingerWebber"

# species,model = "CoCrFeNi","MACE"
# species,model = "NiCoFeCrMn","MACE"
# species,model = "Si","MACE"
# species,model = "AlCoCrFeNi", "MACE"

filename = f"alat-range-finder-{species}-FCC-{model}-10-600.csv"
df = pd.read_csv(filename)


# Define colors, markers, and marker sizes for each order_d
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}

# Create the plot
plt.figure(figsize=(10, 6))

# filter by "good" energy values
min_e = 1e-1 
max_e = 1e+3 
df = df[abs(df['energy']) < max_e]
df = df[abs(df['energy']) > min_e]
df = df.sort_values(by='alat')

# Create the first y-axis
ax1 = plt.gca()  # Get the current axis

# Loop over each unique 'order_d' and plot the corresponding lines
for order in sorted(df['order_d'].unique()):
    subset = df[df['order_d'] == order]

    if subset.empty: continue
    # Plot energy with solid line and different marker
    ax1.plot(subset['alat'], subset['energy'] + abs(subset['energy'].min())  , 
             linestyle='-', color=colors.get(order, 'black'), linewidth=2,
             label=f'Energy order_d={order}',
             )

# Create second y-axis for LED (using twin axis)
ax2 = ax1.twinx()

# Loop again to plot LED values on the second y-axis
for order in sorted(df['order_d'].unique()):
    subset = df[df['order_d'] == order]

    if subset.empty: continue
    # Plot LED with dashed line
    ax2.plot(subset['alat'], abs(subset['led']), 
             linestyle=':', color=colors.get(order, 'black'), linewidth=2,
             label=f'LED order_d={order}')

# Set the y-axes to log scale
# ax1.set_yscale('log')
ax2.set_yscale('log')

# Labeling the axes and the plot
ax1.set_xlabel('alat', fontsize=14)
ax1.set_ylabel('Shifted Energy (log scale)', fontsize=14)
ax2.set_ylabel('LED (log scale)', fontsize=14)
plt.title(f'{model}:Energy and LED vs alat for {species} for first 3 derivatives. \nFiltered for {min_e}eV < |energy| < {max_e}eV', fontsize=14)
plt.axhline(y=1, color='y', linestyle='--', label='LED = 1',linewidth=3,alpha=0.5)


energy_shift = abs(df['energy'].min())
yticks_shifted = ax1.get_yticks()
yticks_original = yticks_shifted - energy_shift
print(yticks_shifted)
print(yticks_original)
ax1.set_yticklabels([f"{val:.2f}" for val in yticks_original])

# Set font size for ticks
plt.xticks(fontsize=12)
ax1.tick_params(axis='y', labelsize=12)  # Set y-axis ticks for energy
ax2.tick_params(axis='y', labelsize=12)  # Set y-axis ticks for LED
plt.yticks(fontsize=12)




for i in subset["energy"]:
    print(i, i + abs(subset['energy'].min()))

print(subset["energy"].min())
print(subset["energy"].max())


# Show the grid (for better readability)
plt.grid(True, which="both", ls="--")

# Create a combined legend for both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)


# Display the plot
plt.show()
