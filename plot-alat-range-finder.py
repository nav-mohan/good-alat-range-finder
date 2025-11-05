input_data = [
    {
        "species": ["Cd", "Te"],
        "model": "SW_WangStroudMarkworth_1989_CdTe__MO_786496821446_001",
        "model_shortname": "SW_WangStroudMarkworth"
    },
    {
        "species": ["Si"],
        "model": "SW_StillingerWeber_1985_Si__MO_405512056662_006",
        "model_shortname": "StillingerWebber"
    },
    {
        "species": ["Si"],
        "model": "SW_LeeHwang_2012GGA_Si__MO_040570764911_001",
        "model_shortname": "SW_LeeHwang"
    },
    {
        "species": ["Al"],
        "model": "EAM_CubicNaturalSpline_ErcolessiAdams_1994_Al__MO_800509458712_003",
        "model_shortname": "EAMCubinNaturalSpline"
    },
    {
        "species": ["Ni", "Co", "Fe", "Cr", "Mn"],
        "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
        "model_shortname": "MACE"
    },
    {
        "species": ["Co", "Cr", "Fe", "Ni"],
        "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
        "model_shortname": "MACE"
    },
    {
        "species": ["Al", "Co", "Cr", "Fe", "Ni"],
        "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
        "model_shortname": "MACE"
    },
    {
        "species": ["Si"],
        "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
        "model_shortname": "MACE"
    },
    {
        "species" : ["H","C","N","O","S"],
        "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
        "model_shortname" : "ALLEGRO"
    },
]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

min_e = 1e-1
max_e = 1e+3

def scientific_notation(x, pos):
    """ Formatter for scientific notation with 2 decimal places """
    return f'{x:.2e}'

for data in input_data:
    model, species, model_shortname = data["model"], data["species"], data["model_shortname"]
    csvfilename = f"alat-range-finder-{''.join(species)}-FCC-{model_shortname}-10-600.csv"
    pngfilename = f"alat-range-finder-{''.join(species)}-FCC-{model_shortname}-10-600.png"
    df = pd.read_csv(csvfilename)
    df = df.sort_values(by='alat')

    # Filter for allowed alat range
    df0 = df[df['order_d'] == 0]
    df0 = df0[(df0['energy'].abs() < max_e) & (df0['energy'].abs() > min_e)]
    allowed_alats = df0['alat'].unique()

    # Define colors
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(19.2 , 10.8))
    axes = axes.flatten()  # Flatten for easier indexing

    unique_orders = sorted(df['order_d'].unique())[:4]  # Take up to 4 orders
    for i, order in enumerate(unique_orders):
        ax1 = axes[i]
        subset = df[df['order_d'] == order]
        subset = subset[subset['alat'].isin(allowed_alats)]
        if subset.empty:
            continue

        # Save original energy for later use
        original_energy = subset['energy'].copy()

        # Check if we need to use log-scale for the energy
        if abs(subset['energy'].max() - subset['energy'].min()) > 1e5:
            # Shift the energy to allow for log-negative values
            subset["energy"] += abs(subset["energy"].min()) + 1
            ax1.set_yscale('log')
            ax1.set_ylabel(f'$\\ln \\partial^{order}E/\\partial a^{order}$ \n shifted', fontsize=12, rotation=0,labelpad=40)
        else:
            ax1.set_ylabel(f'$\\partial^{order}E/\\partial a^{order}$', fontsize=12, rotation=0,labelpad=40)

        # Create twin axis for LED
        ax2 = ax1.twinx()

        # Plot energy
        ax1.plot(subset['alat'], subset['energy'],
                 linestyle='-', color=colors.get(order, 'black'),
                 linewidth=2, label=f'Energy (order_d={order})')

        # Plot LED on secondary axis
        ax2.plot(subset['alat'], abs(subset['led']),
                 linestyle=':', color=colors.get(order, 'black'),
                 linewidth=2, label=f'|LED| (order_d={order})')
        
        if order == 0:
            ax2.plot(subset['alat'], np.ones_like(subset['led']), lw=2, label="LED = 1")

        # Formatting
        ax1.set_title(f'Order {order}', fontsize=13)
        ax1.set_xlabel('alat', fontsize=12)
        ax2.set_ylabel('$\\ln |LED| $', fontsize=12, rotation=0, labelpad=15)
        ax2.set_yscale('log')
        ax1.grid(True, which="both", ls="--")

        # Adjust the y-ticks if log-scale is used for energy
        if ax1.get_yscale() == 'log':
            yticks_shifted = ax1.get_yticks()
            yticks_original = yticks_shifted - (abs(original_energy.min()) + 1)
            ax1.set_yticklabels([f"{val:.2e}" for val in yticks_original])

            # Set the y-axis major formatter to show the exponent only once on the axis
            # ax1.yaxis.set_major_formatter(ScalarFormatter())
            # ax1.yaxis.get_major_formatter().set_powerlimits((0, 0))  # Exponent is shown only once
        # else:
            # If no log-scale, use scientific notation directly
            # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc='upper right')

    # Adjust layout and global title
    fig.suptitle(f'{model_shortname}: Energy and LED vs alat for {species}\nFiltered for {min_e}eV < |energy| < {max_e}eV',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(pngfilename)  # Saves the figure as a PNG file

    # plt.show()
