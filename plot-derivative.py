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

# Load the CSV file
species = "HCNOS"
filename = f"derivative-test-{species}-FCC-ALLEGRO-10-600.csv"
df = pd.read_csv(filename)

# filter by "good" energy values
min_e = 0.1
max_e = 1000

for data in input_data:
    model, species, model_shortname = data["model"], data["species"], data["model_shortname"]
    csvfilename = f"derivative-test-{''.join(species)}-FCC-{model_shortname}-10-600.csv"
    pngfilename = f"derivative-test-{''.join(species)}-FCC-{model_shortname}-10-600.png"
    df = pd.read_csv(csvfilename)
    df = df.sort_values(by='alat')
    
    # filter by energy
    df = df[(df['energy'].abs() < max_e) & (df['energy'].abs() > min_e)]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(19.2, 10.8))

    # Save original energy for later use
    original_energy = df['energy'].copy()

    # Plot energy with markers
    color = 'tab:blue'
    ax1.set_xlabel('alat (Å)', fontsize=14)
    # check if we need to use log-scale for energy
    if abs(df['energy'].max() - df['energy'].min()) > 1e5:
        df['energy'] += abs(df['energy'].min())+1
        ax1.set_yscale('log')
        ax1.set_ylabel('Shifted Energy (log scale)', color=color, fontsize=14, rotation=0, labelpad=15)
    else:
        ax1.set_ylabel('Energy', color=color, fontsize=14, rotation=0, labelpad=15)

    ax1.plot(df['alat'], df['energy'], 
            color=color, linestyle='-',linewidth=4,
            label='Energy'
            )
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)

    # Adjust the y-ticks if log-scale is used for energy
    if ax1.get_yscale() == 'log':
        yticks_shifted = ax1.get_yticks()
        yticks_original = yticks_shifted - (abs(original_energy.min()) + 1)
        ax1.set_yticklabels([f"{val:.2e}" for val in yticks_original])


    # Create second y-axis for errmax
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('$\\ln err $', color=color, fontsize=14, rotation=0, labelpad=15)
    ax2.set_yscale('log')
    ax2.plot(df['alat'], df['errmax'], 
            color=color, linestyle=':', linewidth=4,
            label='errmax')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

    # Title and grid
    
    plt.title(f'{model_shortname}: Plot Energy and errmax vs alat for {species} FCC\nFiltered for {min_e}eV < |energy| < {max_e}eV',fontsize=14)
    if ax1.get_yscale() == 'log':
        plt.title(f'{model_shortname}: Log Plot of Shifted-Energy and errmax vs alat for {species} FCC\nFiltered for {min_e}eV < |energy| < {max_e}eV',fontsize=14)
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

    fig.savefig(pngfilename)  # Saves the figure as a PNG file
    # plt.show()
