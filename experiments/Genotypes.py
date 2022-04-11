import random
import numpy as np
import subprocess as sub
from functools import partial
import os
import sys
import math
from pymoo.core.problem import Problem
from Constants import *
# sys.path.append(os.getcwd() + "/../..")
from evosoro.networks import CPPN
from evosoro.softbot import Genotype, Phenotype
from evosoro.tools.utils import count_occurrences, make_material_tree, rescaled_positive_sigmoid
from evosoro.tools.logging import PrintLog


# Here we are going to evolve the stiffness distribution: need to define min and max elastic modulus
MIN_ELASTIC_MOD = 0.01e6  # when, evolving stiffness, min elastic mod
MAX_ELASTIC_MOD = 1e6  # when, evolving stiffness, max elastic mod
MAX_FREQUENCY = 4.0  # We also evolve a global actuation frequency, max frequency

def frequency_func(x):
    return MAX_FREQUENCY * 2.5 / (np.mean(1/x) + 1.5)  # SAM: inverse the additional inverse in read_write_voxelyze.py

# Defining a custom genotype, inheriting from base class Genotype
class SimpleGenotypeIndirect(Genotype):
    def __init__(self):
        # We instantiate a new genotype for each individual which must have the following properties
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # The genotype consists of a single Compositional Pattern Producing Network (CPPN),
        # with multiple inter-dependent outputs determining the material constituting each voxel
        # (e.g. two types of active voxels, actuated with a different phase, two types of passive voxels, softer and stiffer)
        # The material IDs that you will see in the phenotype mapping dependencies refer to a predefined palette of materials
        # currently hardcoded in tools/read_write_voxelyze.py:
        # (0: empty, 1: passiveSoft, 2: passiveHard, 3: active+, 4:active-),
        # but this can be changed.
        self.add_network(CPPN(output_node_names=["shape", "muscleOrTissue", "muscleType", "tissueType"]))

        self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
                                          dependency_order=["shape", "muscleOrTissue", "muscleType", "tissueType"], output_type=int)  # BUGFIX: "tissueType" was not listed

        self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
                                                        material_if_true=None, material_if_false="0")

        self.to_phenotype_mapping.add_output_dependency(name="muscleOrTissue", dependency_name="shape",
                                                        requirement=True, material_if_true=None, material_if_false=None)  # BUGFIX: was material_if_false=1

        self.to_phenotype_mapping.add_output_dependency(name="tissueType", dependency_name="muscleOrTissue",
                                                        requirement=False, material_if_true="1", material_if_false="2")

        self.to_phenotype_mapping.add_output_dependency(name="muscleType", dependency_name="muscleOrTissue",
                                                        requirement=True, material_if_true="3", material_if_false="4")


# Define a custom phenotype, inheriting from the Phenotype class
class SimplePhenotypeIndirect(Phenotype):
    def is_valid(self, min_percent_full=0.05, max_percent_full = 0.9, min_percent_muscle=0.1, max_percent_muscle = 0.8):
        # override super class function to redefine what constitutes a valid individuals
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():
                return False
            if name == "material":
                state = details["state"]
                # Discarding the robot if it doesn't have at least a given percentage of non-empty voxels
                voxels = np.sum(state>0)
                if voxels < self.genotype.ds_size * min_percent_full or voxels > self.genotype.ds_size * max_percent_full:
                    return False
                # Discarding the robot if it doesn't have at least a given percentage of muscles (materials 3 and 4)
                muscles = count_occurrences(state, [3, 4]) 
                if muscles < voxels * min_percent_muscle or muscles > voxels * max_percent_muscle:
                    return False
        return True


# Defining a custom genotype, inheriting from base class Genotype
class BodyBrainGenotypeIndirect(Genotype):
    def __init__(self):

        # We instantiate a new genotype for each individual which must have the following properties
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # Let's add a first CPPN to the genotype. It dictates a continuous phenotypic trait,
        # the actuation phase of each voxel with respect to a global CPG-like sinusoidal signal
        self.add_network(CPPN(output_node_names=["phase_offset", "frequency"]))

        # Let's map this CPPN output to a VXA tag named <PhaseOffset>
        self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>", 
                                          func=partial(rescaled_positive_sigmoid, x_min=0, x_max=2*math.pi))

        self.to_phenotype_mapping.add_map(name="frequency", tag="<TempPeriod>", env_kws={"frequency": frequency_func})  # tag actually doesn't do anything here

        # Now adding a second CPPN, with three outputs. "shape" the geometry of the robot
        # (i.e. whether a particular voxel is empty or full),
        # if full, "muscleOrTissue" dictates whether a voxel is actuated or passive. The third output, "stiffness",
        # is another continuous attribute, the stiffness (elastic modulus) of each voxel
        # (overrides elastic mod defined in the materials palette)
        self.add_network(CPPN(output_node_names=["shape", "muscleOrTissue"]))

        # Once remapped from [-1,1] to [MIN_ELASTIC_MOD, MAX_ELASTIC_MOD] through "func",
        # the "stiffness" output goes directly to the <Stiffness> vxa tag as a continuous property.
        # We also pass min and max elastic mod as sub-tags of <Stiffness> (will be used by VoxCad)
        # self.to_phenotype_mapping.add_map(name="stiffness", tag="<Stiffness>",
        #                                   func=partial(rescaled_positive_sigmoid, x_min=MIN_ELASTIC_MOD, x_max=MAX_ELASTIC_MOD),
        #                                   params=[MIN_ELASTIC_MOD, MAX_ELASTIC_MOD],
        #                                   param_tags=["MinElasticMod", "MaxElasticMod"])

        # The mapping for materials depends on both "shape" and "muscleOrTissue", through the following dependencies.
        # Basically, if "shape" is false (cppn output < 0), the material with id "0" is assigned (-> empty voxel).
        # If, instead, "shape" is true (cppn output > 0), we look at the "muscleOrTissue" output to determine
        # whether the material is actuated (id = 3) or passive (id = 1). These material IDs are refer to a
        # fixed palette of materials, currently hardcoded in tools/read_write_voxelyze.py
        self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
                                          dependency_order=["shape", "muscleOrTissue"], output_type=int)

        self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
                                                        material_if_true=None, material_if_false="0")

        self.to_phenotype_mapping.add_output_dependency(name="muscleOrTissue", dependency_name="shape",
                                                        requirement=True, material_if_true="3", material_if_false="1")



# Defining a custom genotype, inheriting from base class Genotype
class BodyBrainGenotypeIndirect2(Genotype):
    def __init__(self):

        # We instantiate a new genotype for each individual which must have the following properties
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # Let's add a first CPPN to the genotype. It dictates a continuous phenotypic trait,
        # the actuation phase of each voxel with respect to a global CPG-like sinusoidal signal
        self.add_network(CPPN(output_node_names=["phase_offset", "frequency"]))

        # Let's map this CPPN output to a VXA tag named <PhaseOffset>
        self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>", 
                                          func=partial(rescaled_positive_sigmoid, x_min=0, x_max=2*math.pi))

        self.to_phenotype_mapping.add_map(name="frequency", tag="<TempPeriod>", env_kws={"frequency": frequency_func})  # tag actually doesn't do anything here

        # The genotype consists of a single Compositional Pattern Producing Network (CPPN),
        # with multiple inter-dependent outputs determining the material constituting each voxel
        # (e.g. two types of active voxels, actuated with a different phase, two types of passive voxels, softer and stiffer)
        # The material IDs that you will see in the phenotype mapping dependencies refer to a predefined palette of materials
        # currently hardcoded in tools/read_write_voxelyze.py:
        # (0: empty, 1: passiveSoft, 2: passiveHard, 3: active+, 4:active-),
        # but this can be changed.
        self.add_network(CPPN(output_node_names=["shape", "muscleOrTissue", "muscleType", "tissueType"]))

        self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
                                          dependency_order=["shape", "muscleOrTissue", "muscleType", "tissueType"], output_type=int)  # BUGFIX: "tissueType" was not listed

        self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
                                                        material_if_true=None, material_if_false="0")

        self.to_phenotype_mapping.add_output_dependency(name="muscleOrTissue", dependency_name="shape",
                                                        requirement=True, material_if_true=None, material_if_false=None)  # BUGFIX: was material_if_false=1

        self.to_phenotype_mapping.add_output_dependency(name="tissueType", dependency_name="muscleOrTissue",
                                                        requirement=False, material_if_true="1", material_if_false="2")

        self.to_phenotype_mapping.add_output_dependency(name="muscleType", dependency_name="muscleOrTissue",
                                                        requirement=True, material_if_true="3", material_if_false="4")