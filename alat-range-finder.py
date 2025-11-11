
input_data = [
    # {
    #     "species": ["Cd", "Te"],
    #     "model": "SW_WangStroudMarkworth_1989_CdTe__MO_786496821446_001",
    #     "model_shortname": "SW_WangStroudMarkworth"
    # },
    # {
    #     "species": ["Si"],
    #     "model": "SW_StillingerWeber_1985_Si__MO_405512056662_006",
    #     "model_shortname": "StillingerWebber"
    # },
    # {
    #     "species": ["Si"],
    #     "model": "SW_LeeHwang_2012GGA_Si__MO_040570764911_001",
    #     "model_shortname": "SW_LeeHwang"
    # },
    # {
    #     "species": ["Al"],
    #     "model": "EAM_CubicNaturalSpline_ErcolessiAdams_1994_Al__MO_800509458712_003",
    #     "model_shortname": "EAMCubinNaturalSpline"
    # },
    # {
    #     "species": ["Ni", "Co", "Fe", "Cr", "Mn"],
    #     "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
    #     "model_shortname": "MACE"
    # },
    # {
    #     "species": ["Co", "Cr", "Fe", "Ni"],
    #     "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
    #     "model_shortname": "MACE"
    # },
    # {
    #     "species": ["Al", "Co", "Cr", "Fe", "Ni"],
    #     "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
    #     "model_shortname": "MACE"
    # },
    # {
    #     "species": ["Si"],
    #     "model": "TorchML_20231203_MACE_MP_0_128_L1_EP199__MO_000000000000_000",
    #     "model_shortname": "MACE"
    # },
    # {
    #     "species" : ["H","C","N","O","S"],
    #     "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
    #     "model_shortname" : "ALLEGRO"
    # },

    {
        "species" : ["H"],
        "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
        "model_shortname" : "ALLEGRO"
    },
    {
        "species" : ["C"],
        "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
        "model_shortname" : "ALLEGRO"
    },
    {
        "species" : ["N"],
        "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
        "model_shortname" : "ALLEGRO"
    },
    {
        "species" : ["O"],
        "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
        "model_shortname" : "ALLEGRO"
    },
    {
        "species" : ["S"],
        "model": "TorchML_Allegro_NikidisKyriakopoulosTohidKachrimanisKioseoglou_2024_HCNOS__MO_000000000000_000",
        "model_shortname" : "ALLEGRO"
    },
]


# given an energy-vs-alat curve, find the range of alats that are good
# per-atom energies should be bounded within [100mEV and 1000ev]
# energies should be continuous 
# this would make use of the algorithm from Dimer-VC

import numdifftools as nd
import numpy as np
import math

from ase.calculators.kim import KIM, get_model_supported_species
import kim_tools.ase as kim_ase_utils
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic

import sys
sys.path.append('/h/mohan227/allegro-issues/')
from simplelogger import SimpleLogger
from simplecsv import SimpleCSV

# A helper function to find the minimum-cutoff for a given model and a list of species
# it considers each dimer-combination and finds their cutoff and then returns the minimum
XTOL = 1e-4 # 1e-8
ETOL_COARSE=1e-6 # 1e-6
ETOL_FINE=1e-15 #1e-15
MAX_BISECT_ITERS=1000 #1000
MAX_UPPER_CUTOFF_BRACKET=20.0 #20.0
def compute_minimum_cutoff(model:str, species:list)->float:
    cutoffs = []
    for i in range(len(species)):
        si : str = species[i]
        Rii:float = kim_ase_utils.get_model_energy_cutoff(model,[si,si], XTOL, ETOL_COARSE,ETOL_FINE,MAX_BISECT_ITERS,MAX_UPPER_CUTOFF_BRACKET)
        cutoffs.append(Rii)
        for j in range(i+1,len(species)):
            sj : str = species[j]
            try:
                Rij:float = kim_ase_utils.get_model_energy_cutoff(model,[sj,si], XTOL, ETOL_COARSE,ETOL_FINE,MAX_BISECT_ITERS,MAX_UPPER_CUTOFF_BRACKET)
                cutoffs.append(Rij)
            except:
                pass
    return np.min(cutoffs)

# generates an FCC of size alat, with species and computes energy using model
def energy(alat,species,model):
    ncells_per_side = 2
    while True:
        atoms = FaceCenteredCubic(
            size=(ncells_per_side, ncells_per_side, ncells_per_side),
            latticeconstant=alat,
            symbol="H",
            pbc=False,
        )
        if len(atoms) < len(species):
            ncells_per_side += 1
        else:
            break
    kim_ase_utils.randomize_species(atoms, species)
    calc = KIM(model)
    atoms.set_calculator(calc)
    # compute energy
    try:
        # logger.info(f"TRY COMPUTE POTENTIAL ENERGY {spec} | {alat}")
        pe = atoms.get_potential_energy()
        logger.info(f"SUCCESS COMPUTE POTENTIAL ENERGY {','.join(species)} | {alat} | {pe}")
        return pe
    except Exception as e:
        logger.info(e)
        logger.info(f"FAILED COMPUTE POTENTIAL ENERGY {','.join(species)} | {alat}")
        
    pass


for data in input_data:
    model,species,model_shortname = data["model"], data["species"], data["model_shortname"]
    logfilename = f"alat-range-finder-{''.join(species)}-FCC-{model_shortname}-10-600.log"
    csv_outfile = f"alat-range-finder-{''.join(species)}-FCC-{model_shortname}-10-600.csv"
    logger = SimpleLogger(logfilename)
    csvwriter = SimpleCSV(["species", "alat", "energy", "order_d", "led", "status"], csv_outfile)

    min_cutoff = compute_minimum_cutoff(model,species)
    amax = 1.0*min_cutoff
    amin = 0.3*min_cutoff 
    del_a = 0.01
    na = int(math.ceil((amax - amin) / del_a))
    max_deriv = 3 # largest derivative to be investigated
    continuous = [True] * (1 + max_deriv)  # assume function and all

    led_tolerance = 1.0
    for n in range(0,max_deriv+1):
        alats = []
        energies = []

        logger.info(f"COMPUTE DERIVATIVE OF ORDER {n}")
        Denergy = nd.Derivative(energy, step=0.1*del_a, full_output = True, method="backward", n=n)
        
        # generate energy curve
        for j in range(0,na+1):
            a = amin + j * del_a
            try:
                val,info = Denergy(a,species,model)
                alats.append(a)
                energies.append(val.item())
            
            except Exception as e:
                logger.info(f"COMPUTE DERIVATIVE EXCEPTION | ORDER = {n} | j,alat = {j,alats[j]}")
                continue

        fact = 1.0/6.0
        is_continuous = True
        for j in range(2,na-3):
            logger.info(f"CHECK DERIVATIVE OF ORDER {n} | alat,energy = {alats[j],energies[j]}")
            # use 5-th order local difference formula
            led = fact * (
                - energies[j-2] 
                + 5 * energies[j-1]
                - 10 * energies[j]
                + 10 * energies[j+1]
                - 5 * energies[j+2]
                + energies[j+3] 
            )
            if abs(led) > led_tolerance:
                logger.info(f"FAILED DERIVATIVE OF ORDER {n} AT j,alat,energy = {j},{alats[j]},{energies[j]}")
                continuous[n] = False
                is_continuous = False
            else: 
                logger.info(f"SUCCESS DERIVATIVE OF ORDER {n} AT j,alat,energy = {j},{alats[j]},{energies[j]}")
                is_continuous = True
                continuous[n] = True
            
            status = 1 if is_continuous else 0
            rowdata = [",".join(species), alats[j],energies[j], n, led, is_continuous]
            csvwriter.append(rowdata)
            
        if is_continuous:
            logger.info(f"NO DISCONTINUITIES FOUND IN DERIVATIVE OF ORDER {n}")
        else:
            logger.info(f"DISCONTINUITIES FOUND IN DERIVATIVE OF ORDER {n}")

