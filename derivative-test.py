
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

# check the derivative test for a specific species at a specific alat

from numdifftools.step_generators import MaxStepGenerator
import numdifftools as nd
import numpy as np

import ase
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.kim.kim import KIM, get_model_supported_species
import kim_tools.ase as kim_ase_utils
from ase import Atoms

from ase.build import bulk
from ase import io

from itertools import combinations

# from covalent import COVALENT_RADII,create_lattice_with_covalent_spacing

# --- Log File, CSV File ---
import sys
sys.path.append('/h/mohan227/allegro-issues/')
from simplelogger import SimpleLogger
from simplecsv import SimpleCSV


def negpot(p, at=0, dof=0, atoms=None):
    """
    Function that takes the value 'p' of degree of freedom 'dof' of atom 'at'
    and returns the negative of the total potential energy of full system of
    atoms. Used by the numerical derivative method.
    """
    # logger.info(f"negpot ARGS: | p = {p} | at = {at} | dof = {dof} | atoms = {atoms}")
    if atoms is None:
        return 0
    sve = (atoms[at].position)[dof]
    (atoms[at].position)[dof] = p
    pot = atoms.get_potential_energy()
    (atoms[at].position)[dof] = sve
    # logger.info(f"negpot RETURNS {-pot}")
    return -pot

def perform_numerical_derivative_check(atoms, heading, dashwidth):
    logger.info(f"perform_numerical_derivative_check atoms = {atoms}")
    """
    Perform a numerical derivative check for the ASE atoms object in 'atoms'.
    """
    # compute analytical forces (negative gradient of cohesive energy)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # Loop over atoms and compute numerical derivative check
    sg = [
        MaxStepGenerator(
            base_step=1e-4, num_steps=14, use_exact_steps=True, step_ratio=1.6, offset=0
        ),
        MaxStepGenerator(
            base_step=1e-3, num_steps=14, use_exact_steps=True, step_ratio=1.6, offset=0
        ),
        MaxStepGenerator(
            base_step=1e-2, num_steps=14, use_exact_steps=True, step_ratio=1.6, offset=0
        ),
    ]
    nsg = 4  # number of step generators to try
    Dnegpot = [nd.Derivative(negpot, full_output=True)]
    for i in range(nsg - 1):
        Dnegpot.append(nd.Derivative(negpot, step=sg[i], full_output=True))
    forces_num = np.zeros(shape=(len(atoms), 3), dtype=float, order="C")
    forces_uncert = np.zeros(shape=(len(atoms), 3), dtype=float, order="C")
    forces_failed = np.zeros(shape=(len(atoms), 3), dtype=int, order="C")
    for at in range(0, len(atoms)):
        for dof in range(0, 3):
            p = (atoms[at].position)[dof]
            errmin_sg = 1e30
            failed_to_get_deriv = True
            for i in range(nsg):
                try:
                    # logger.info(f"TRY compute derivative p = {p} | at = {at} | dof = {dof} | atoms = {atoms}")
                    val, info = Dnegpot[i](p, at=at, dof=dof, atoms=atoms)
                    # logger.info(f"SUCCESS compute derivative p = {p} | at = {at} | dof = {dof} | atoms = {atoms}")
                    if abs(val - forces[at, dof]) < errmin_sg:
                        errmin_sg = abs(val - forces[at, dof])
                        val_best = val
                        info_best = info
                        failed_to_get_deriv = False
                except:  # noqa: E722
                    # logger.info(f"EXCEPT compute derivative p = {p} | at = {at} | dof = {dof} | atoms = {atoms}")
                    # Failed to compute derivative, so skip this value
                    (atoms[at].position)[dof] = p  # Restore value which may have been
                    # left changed when exception was
                    # generated.
            if failed_to_get_deriv:
                # Failed all attempts for this at/dof, assume this is
                # because the potential is being evaluated outside its
                # legal range and skip. (TODO: A more careful check
                # on the reason for the error would be good.)
                forces_failed[at, dof] = 1
            else:
                val = val_best
                info = info_best
                forces_num[at, dof] = val
                forces_uncert[at, dof] = info.error_estimate

    # Identify outliers using a box plot construction with fences
    # (See http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm)
    # We'll take all results above the upper outer fence to be outliers.
    #
    # create flattened version of forces_uncert without failures
    numterms = 3 * len(atoms) - np.sum(forces_failed)
    forces_uncert_without_failures = np.zeros(numterms, dtype=float)
    i = -1
    for at in range(0, len(atoms)):
        for dof in range(0, 3):
            if not forces_failed[at, dof]:
                i += 1
                forces_uncert_without_failures[i] = forces_uncert[at, dof]
    
    logger.info(f"COMPUTING QUARTILE FOR {forces_uncert_without_failures}")
    uncert_lower_quartile = np.percentile(forces_uncert_without_failures, 25)
    uncert_upper_quartile = np.percentile(forces_uncert_without_failures, 75)
    uncert_interquartile_range = uncert_upper_quartile - uncert_lower_quartile
    uncert_upper_fence = uncert_upper_quartile + 3 * uncert_interquartile_range

    # Initialize for printing
    frmt_head = "{0:>6}  {1:>4} {2:>3} {3:>25} {4:>25} {5:>15} {6:>15}"
    frmt_line = {
        True: "{0: 6d}  {1:4} {2: 3d} {3: 25.15e} {4: 25.15e} {5: 15.5e} {6: 15.5e} {7:1}",
        False: "             {2: 3d} {3: 25.15e} {4: 25.15e} {5: 15.5e} {6: 15.5e} {7:1}",
    }
    frmt_fail = "{0: 6d}  {1:4} {2: 3d} {3:>25} {4:>25} {5:>15} {6:>15} {7:1}"
    logger.info(
        "Comparison of analytical forces obtained from the model, "
        "the force computed as a numerical derivative"
    )
    logger.info(
        "of the energy, the difference between them, and the uncertainty "
        "in the numerical estimate of the force."
    )
    logger.info(
        "The computed equilibrium lattice constant for the crystal (a0) "
        "is given in the heading."
    )
    logger.info("")
    logger.info(heading)
    logger.info("-" * dashwidth)
    args = (
        "Part",
        "Spec",
        "Dir",
        "Force_model",
        "Force_numer",
        "|Force diff|",
        "uncertainty",
    )
    logger.info(frmt_head.format(*args))
    logger.info("-" * dashwidth)

    # Identify max error and print numerical derivative results
    eps_prec = np.finfo(float).eps
    errmax = 0.0
    at_least_one_result_discarded = False
    at_least_one_failure = False
    for at in range(0, len(atoms)):
        for dof in range(0, 3):
            if forces_failed[at, dof]:
                # skip at/dof for which force was not computed
                args = (at + 1, atoms[at].symbol, dof + 1, "", "", "", "", "F")
                logger.info(frmt_fail.format(*args))
                at_least_one_failure = True
                continue
            forcediff = abs(forces[at, dof] - forces_num[at, dof])
            den = max(abs(forces_num[at, dof]), eps_prec)
            if forces_uncert[at, dof] < uncert_upper_fence:
                # Result is not an outlier. Include it in determining max error
                lowacc_mark = " "
                if forcediff / den > errmax:
                    errmax = forcediff / den
                    at_max = at
                    dof_max = dof
            else:
                lowacc_mark = "*"
                at_least_one_result_discarded = True
            # Print results line
            args = (
                at + 1,
                atoms[at].symbol,
                dof + 1,
                forces[at, dof],
                forces_num[at, dof],
                forcediff,
                forces_uncert[at, dof],
                lowacc_mark,
            )
            logger.info(frmt_line[dof == 0].format(*args))
            if dof == 2:
                logger.info("-" * dashwidth)
    if at_least_one_result_discarded:
        logger.info(
            "* Starred lines are suspected outliers and are not "
            "included when determining the error."
        )
        logger.info(
            "  A calculation is considered an outlier if it has an "
            "uncertainty that lies at an abnormal"
        )
        logger.info(
            "  distance from the other uncertainties in this set of "
            "calculations.  Outliers are determined"
        )
        logger.info(
            "  using the box plot construction with fences. "
            "An outlier could indicate a problem with the"
        )
        logger.info(
            "  the numerical differentiation or problems with the "
            "potential energy, such as discontinuities."
        )
    if at_least_one_failure:
        if at_least_one_result_discarded:
            logger.info("")
        logger.info(
            "WARNING: Numerical derivative could not be computed for "
            "at least one atom/dof."
        )
        logger.info(
            '         Failed calculations indicated with an "F". This '
            "can be due to attempts"
        )
        logger.info(
            "         to evaluate the model outside its legal range "
            "during a numerical derivative."
        )
        logger.info(
            "         calculation. These lines are ignored when " "computing the error."
        )

    # Print summary
    logger.info("")
    logger.info(
        "Maximum error obtained for particle = {0:d}, direction = {1:d}:".format(
            at_max + 1, dof_max + 1
        )
    )
    logger.info("")
    logger.info("              |F_model - F_numer|")
    logger.info("    error = ----------------------- = {0:.5e}".format(errmax))
    logger.info("              max{|F_numer|, eps}")
    logger.info("")
    logger.info("")
    
    return errmax

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



for data in input_data:
    model,species,model_shortname = data["model"], data["species"], data["model_shortname"]
    calc = KIM(model)
    logfilename = f"derivative-test-{''.join(species)}-FCC-{model_shortname}-10-600.log"
    csv_outfile = f"derivative-test-{''.join(species)}-FCC-{model_shortname}-10-600.csv"
    logger = SimpleLogger(logfilename)
    csvwriter = SimpleCSV(["species", "alat", "energy", "status", "errmax"], csv_outfile)

    min_cutoff = compute_minimum_cutoff(model,species)
    amax = 1.0*min_cutoff
    amin = 0.3*min_cutoff 
    ncells_per_side = 2
    for alat in np.linspace(amin,amax, 101):
        spec = "".join(species)
        data = [spec, alat]
        while True:
            atoms = FaceCenteredCubic(
                size = (ncells_per_side,ncells_per_side,ncells_per_side),
                latticeconstant=alat,
                symbol = "H",
                pbc = False
            )
            if(len(atoms) < len(species)):
                ncells_per_side += 1
            else: 
                break
        kim_ase_utils.randomize_species(atoms,species)
        atoms.set_calculator(calc)

        # compute energy
        try:
            # logger.info(f"TRY COMPUTE POTENTIAL ENERGY {spec} | {alat}")
            pe = atoms.get_potential_energy()
            # logger.info(f"SUCCESS COMPUTE POTENTIAL ENERGY {spec} | {alat} | {pe}")
            data.append(pe)
        except Exception as e:
            logger.info(e)
            logger.info(f"FAILED COMPUTE POTENTIAL ENERGY {spec} | {alat}")
            continue

        # Perform Derivative Check

        large_cell_len = 7 * alat * ncells_per_side
        atoms.set_cell([large_cell_len, large_cell_len, large_cell_len])
        trans = [0.5 * large_cell_len] * 3
        atoms.translate(trans)

        print("alat: ", alat)
        print("ncells_per_side: ", ncells_per_side)
        print("N atoms:", len(atoms))
        print("Formula:", atoms.get_chemical_formula())
        print("Cell vectors (Å):\n", atoms.get_cell())
        print("Positions (min,max) Å:", atoms.get_positions().min(axis=0), atoms.get_positions().max(axis=0))
        print("Periodic boundary flags:", atoms.get_pbc())
        print("------------------------------")

        aux_file = "config-" + spec + ".xyz"
        a0string = "a0 = {}".format(alat)
        heading = (
            "MONOATOMIC STRUCTURE -- Species = "
            + spec
            + " | "
            + a0string
            + ' | (Configuration in file "'
            + aux_file
            + '")'
        )
        dashwidth = 100
        try:
            logger.info(f"START RESCALE {spec} | {alat}")
            kim_ase_utils.rescale_to_get_nonzero_forces(atoms, 0.01)
            logger.info(f"START RANDOMIZE {spec} | {alat}")
            kim_ase_utils.randomize_positions(atoms, 0.1*alat)
            logger.info(f"START PERTURB {spec} | {alat}")
            kim_ase_utils.perturb_until_all_forces_sizeable(atoms, 0.1*alat)
            logger.info(f"START FND {spec} | {alat}")
            errmax = perform_numerical_derivative_check(atoms, heading, dashwidth)
            logger.info(f"SUCCESS FORCES_NUM_DER CHECK {spec} | {alat} | errmax = {errmax}")
            data.append("1")
            data.append(errmax)
        except Exception as e:
            logger.info(f"FAILED FORCES_NUM_DER CHECK {spec} | {alat} | {e}")
            data.append("0")
            data.append("-1")
        
        csvwriter.append(data)

        # write out lattice configuration XYZ file
        # dumpfilename = f"/scratch/mohan227/dump-pe-alat-{spec}-{alat}.xyz"
        # io.write(dumpfilename,atoms)

