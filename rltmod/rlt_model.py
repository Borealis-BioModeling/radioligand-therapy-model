"""
Semi-mechanistic model of radioligand therapy (RLT).
"""

from pysb import *
from pysb import macros
import pysb.units
from pysb.units import units
from pysb.pkpd import macros as pkpd

pysb.units.add_macro_units(pkpd)
pysb.units.add_macro_units(macros)

import numpy as np

LN2 = np.log(2.0)

with units():
    ##  Initialize the Model ##
    Model()

    ##  Core Simulation Units  ##
    SimulationUnits(time="h", concentration="nM", volume="L")

    ##  Compartments  ##
    Parameter("V_CENTRAL", 15.0, unit="L")
    Parameter("V_PERIPHERAL", 10.0, unit="L")
    Parameter("V_CANCER", 0.5, unit="L")
    pkpd.three_compartments(
        "CENTRAL", V_CENTRAL, "PERIPHERAL", V_PERIPHERAL, "CANCER", V_CANCER
    )

    # Radioligand therapy
    #  decay: F = radioactive, T = radioisotope has decayed
    #  endo: F = extracellular, T = taken inside a cell by endocytosis
    Monomer("RLT", ["b", "decay", "endo"], {"decay": ["T", "F"], "endo": ["T", "F"]})
    Parameter("dose_RLT_CENTRAL", 100.0, unit="nmol")
    pkpd.dose_bolus(RLT(b=None, decay="F", endo="F"), CENTRAL, dose_RLT_CENTRAL)
    Parameter("dose_RLT_decay_CENTRAL", 0.0, unit="nmol")
    Expression("expr_dose_RLT_decay_CENTRAL", dose_RLT_decay_CENTRAL / V_CENTRAL)
    Initial(RLT(b=None, decay="T", endo="F") ** CENTRAL, expr_dose_RLT_decay_CENTRAL)

    # Target Biomarker
    Monomer("Biomarker", ["b", "endo"], {"endo": ["T", "F"]})
    Parameter("Biomarker_0", 100.0, unit="nM")
    Initial(Biomarker(b=None, endo="F") ** CANCER, Biomarker_0)

    # RLT decay
    Parameter("rlt_halflife", 2.67, unit="day")  # Yttrium-90 t_1/2 = 2.67 days
    Expression("k_rlt_decay", LN2 / rlt_halflife)
    # Define an Emission monomer to monitor the amount of radioactive emissions generated in each compartment and under different conditions.
    Monomer("Emission", ["endo", "bound"], {"endo": ["F", "T"], "bound": ["F", "T"]})
    # Defining multiple rules here to track the state of Emissions, whehter they
    # are just in the compartment versus bound to biomarker versus internalized
    # by endocytic processes.
    Rule(
        "rlt_decay_central",
        RLT(b=WILD, decay="F") ** CENTRAL
        >> RLT(b=WILD, decay="T") ** CENTRAL + Emission(endo="F", bound="F") ** CENTRAL,
        k_rlt_decay,
    )
    Rule(
        "rlt_decay_peripheral",
        RLT(b=WILD, decay="F") ** PERIPHERAL
        >> RLT(b=WILD, decay="T") ** PERIPHERAL
        + Emission(endo="F", bound="F") ** PERIPHERAL,
        k_rlt_decay,
    )
    Rule(
        "rlt_decay_cancer_free",
        RLT(b=None, decay="F", endo="F") ** CANCER
        >> RLT(b=None, decay="T", endo="F") ** CANCER
        + Emission(endo="F", bound="F") ** CANCER,
        k_rlt_decay,
    )
    Rule(
        "rlt_decay_cancer_bound",
        RLT(b=ANY, decay="F", endo="F") ** CANCER
        >> RLT(b=ANY, decay="T", endo="F") ** CANCER
        + Emission(endo="F", bound="T") ** CANCER,
        k_rlt_decay,
    )
    Rule(
        "rlt_decay_cancer_bound_endo",
        RLT(b=ANY, decay="F", endo="T") ** CANCER
        >> RLT(b=ANY, decay="T", endo="T") ** CANCER
        + Emission(endo="T", bound="T") ** CANCER,
        k_rlt_decay,
    )
    # Leaving the generated Emission as endo=T and bound=T since
    # it was generated after binding.
    Rule(
        "rlt_decay_cancer_endo",
        RLT(b=None, decay="F", endo="T") ** CANCER
        >> RLT(b=None, decay="T", endo="T") ** CANCER
        + Emission(endo="T", bound="T") ** CANCER,
        k_rlt_decay,
    )
    ##  Drug Distribution  ##
    # CENTRAL <-> PERIPHERAL
    Parameter("kf_distribute_RLT_CENTRAL_PERIPHERAL", 0.1, unit=" 1 / h")
    Parameter("kr_distribute_RLT_CENTRAL_PERIPHERAL", 0.1, unit=" 1 / h")
    pkpd.distribute(
        RLT(b=None, endo="F"),
        CENTRAL,
        PERIPHERAL,
        klist=[
            kf_distribute_RLT_CENTRAL_PERIPHERAL,
            kr_distribute_RLT_CENTRAL_PERIPHERAL,
        ],
    )
    # CENTRAL <-> CANCER
    Parameter("kf_distribute_RLT_CENTRAL_CANCER", 0.05, unit=" 1 / h")
    Parameter("kr_distribute_RLT_CENTRAL_CANCER", 0.05, unit=" 1 / h")
    pkpd.distribute(
        RLT(b=None, endo="F"),
        CENTRAL,
        CANCER,
        klist=[kf_distribute_RLT_CENTRAL_CANCER, kr_distribute_RLT_CENTRAL_CANCER],
    )

    ##  Biomarker Binding  ##
    Parameter("kr_bind_rlt_biomarker", 1e-3, unit="1/s")
    Parameter("Kd_rlt_biomarker", 10.0, unit="nM")
    Expression("kf_bind_rlt_biomarker", kr_bind_rlt_biomarker / Kd_rlt_biomarker)
    # Binding between extracellular RLT (endo="F") and membrane-bound biomarker (endo="F")
    macros.bind(
        RLT(endo="F") ** CANCER,
        "b",
        Biomarker(endo="F") ** CANCER,
        "b",
        [kf_bind_rlt_biomarker, kr_bind_rlt_biomarker],
    )

    ## Endocytosis of bound biomarker ##
    Parameter("k_endo", 1e-4, unit="1/s")
    Rule(
        "biomarker_endocytosis",
        RLT(b=ANY, endo="F") % Biomarker(b=ANY, endo="F") ** CANCER
        >> RLT(b=ANY, endo="T") % Biomarker(b=ANY, endo="T") ** CANCER,
        k_endo,
    )

    ## Biomarker unbinding after endocytosis
    Parameter("k_unbind", 1e-2, unit="1 / s")
    Rule(
        "endocytic_unbinding",
        RLT(b=1, endo="T") % Biomarker(b=1, endo="T") ** CANCER
        >> RLT(b=None, endo="T") + Biomarker(b=None, endo="T") ** CANCER,
        k_unbind,
    )

    ## Biomarker recycling
    Parameter("k_biomarker_recycle", 1e-4, unit="1 / s")
    Rule(
        "biomarker_recyle",
        Biomarker(b=None, endo="T") ** CANCER >> Biomarker(b=None, endo="F") ** CANCER,
        k_biomarker_recycle,
    )

    ##  Drug Elimination  ##
    Parameter("kel_RLT_CENTRAL", 0.1, unit=" 1 / h")
    pkpd.eliminate(RLT, CENTRAL, kel_RLT_CENTRAL)

    ##  Observables  ##
    # Total RLT in each compartment
    Observable("obs_RLT_CENTRAL", RLT() ** CENTRAL)
    Observable("obs_RLT_PERIPHERAL", RLT() ** PERIPHERAL)
    Observable("obs_RLT_CANCER", RLT(b=WILD, endo="F") ** CANCER)
    # Radioactive RLT in each compartment
    Observable("obs_RLT_CENTRAL_active", RLT(b=None, decay="F", endo="F") ** CENTRAL)
    Observable(
        "obs_RLT_PERIPHERAL_active", RLT(b=None, decay="F", endo="F") ** PERIPHERAL
    )
    Observable("obs_RLT_CANCER_active", RLT(b=WILD, decay="F") ** CANCER)

    # Decayed RLT in each compartment
    Observable("obs_RLT_CENTRAL_decay", RLT(b=None, decay="T") ** CENTRAL)
    Observable("obs_RLT_PERIPHERAL_decay", RLT(b=None, decay="T") ** PERIPHERAL)
    Observable("obs_RLT_CANCER_decay_exo", RLT(b=WILD, decay="T", endo="F") ** CANCER)

    # Emissions in each compartment and state
    # All emissions
    Observable("obs_Emissions", Emission())
    # Emissions in the CENTRAL compartment
    Observable("obs_Emission_CENTRAL", Emission() ** CENTRAL)
    # Emissions in the PERIPHERAL compartment
    Observable("obs_Emission_PERIPHERAL", Emission() ** PERIPHERAL)
    Expression(
        "obs_Emission_off_target",
        (
            (obs_Emission_CENTRAL * V_CENTRAL)
            + (obs_Emission_PERIPHERAL * V_PERIPHERAL) / (V_CENTRAL + V_PERIPHERAL)
        ),
    )
    # Emissions in the CANCER compartment
    Observable("obs_Emission_CANCER", Emission() ** CANCER)
    # Emissions in the CANCER compartment from RLT bound to the biomarker
    Observable("obs_Emission_CANCER_bound", Emission(bound="T", endo="F") ** CANCER)
    # Emissions in the CANCER compartment from RLT bound to biomarker that
    # has undergone endocytosis or from RLT that was endocytosed but unbound
    # from the biomarker.
    Observable(
        "obs_Emission_CANCER_bound_endo", Emission(endo="T", bound="T") ** CANCER
    )
    # Emissions in the CANCER compartment from RLT bound to biomarker that
    # has undergone endocytosis
    Expression(
        "obs_Targeted_Emissions",
        (obs_Emission_CANCER_bound + obs_Emission_CANCER_bound_endo),
    )
    # Biomarker
    Observable("obs_exo_Biomarker", Biomarker(endo="F") ** CANCER)
    Observable("obs_endo_Biomarker", Biomarker(endo="T") ** CANCER)
