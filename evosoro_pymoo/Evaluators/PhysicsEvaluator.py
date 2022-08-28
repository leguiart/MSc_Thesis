import json
import os
import sys
import time
import logging
import numpy as np
import subprocess as sub
from lxml import etree

#sys.path.append(os.getcwd() + "/../..")
from evosoro.tools.read_write_voxelyze import read_voxlyze_results, write_voxelyze_file, get_vxd
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from common.Utils import readFromJson, readFromPickle, saveToPickle, timeit, writeToJson
from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from evosoro_pymoo.common.IRecoverFromFile import IFileRecovery
from evosoro_pymoo.common.IStart import IStarter

logger = logging.getLogger(f"__main__.{__name__}")


def folder_heirarchy_creation_helper(run_directory, save_networks, save_all_individual_data, save_lineages, resuming_run = False):
    # clear directory
    if not resuming_run:
        sub.call("rm -rf " + run_directory + "/* 2>/dev/null", shell=True)

        sub.call("mkdir " + run_directory + "/voxelyzeFiles 2> /dev/null", shell=True)
        sub.call("mkdir " + run_directory + "/tempFiles 2> /dev/null", shell=True)
        sub.call("mkdir " + run_directory + "/fitnessFiles 2> /dev/null", shell=True)

        sub.call("mkdir " + run_directory + "/bestSoFar 2> /dev/null", shell=True)
        sub.call("mkdir " + run_directory + "/bestSoFar/paretoFronts 2> /dev/null", shell=True)
        sub.call("mkdir " + run_directory + "/bestSoFar/fitOnly 2>/dev/null", shell=True)

        sub.call("mkdir " + run_directory + "/pickledPops 2> /dev/null", shell=True)

        if save_all_individual_data:
            sub.call("mkdir " + run_directory + "/allIndividualsData", shell=True)
            sub.call("rm -f " + run_directory + "/allIndividualsData/* 2>/dev/null", shell=True)  # TODO: why clear these

        if save_networks:
            sub.call("mkdir " + run_directory + "/network_gml", shell=True)
            sub.call("rm -rf " + run_directory + "/network_gml/* 2>/dev/null", shell=True)

        if save_lineages:
            sub.call("mkdir " + run_directory + "/ancestors 2> /dev/null", shell=True)


def initialize_folder_heirarchy(run_directory, save_networks, save_all_individual_data=True,
                       save_lineages=True, resuming_run = False):

    if os.path.exists(run_directory) and os.path.isdir(run_directory):

        folder_heirarchy_creation_helper(run_directory, save_networks, 
                                        save_all_individual_data, save_lineages, 
                                        resuming_run)
    else:
        sub.call("mkdir " + run_directory + "/" + " 2>/dev/null", shell=True)
        folder_heirarchy_creation_helper(run_directory, save_networks, 
                                save_all_individual_data, save_lineages)



def create_gen_directories(gen, run_directory, save_vxa_every, save_networks):

    print ("\n\n")
    print ("----------------------------------")
    print ("---------- GENERATION", gen, "----------")
    print ("----------------------------------")
    print ("\n")
    sub.call("rm -rf " + run_directory + "/Gen_%04i" % gen, shell=True)
    if gen % save_vxa_every == 0 and save_vxa_every > 0:
        sub.call("mkdir " + run_directory + "/Gen_%04i" % gen, shell=True)

    if save_networks:
        sub.call("mkdir " + run_directory + "/network_gml/Gen_%04i" % gen, shell=True)


class BaseSoftBotPhysicsEvaluator(IEvaluator, ICheckpoint, IStarter, IFileRecovery):
    def __init__(self, sim, env, save_vxa_every, run_directory, run_name, 
                objective_dict, max_gens, num_env_cycles = 0, max_eval_time=60, 
                time_to_try_again=10, save_lineages = True, save_nets = False):
        """Evaluate all individuals of the population in VoxCad.

        Parameters
        ----------
        sim : Sim
            Configures parameters of the Voxelyze simulation.

        env : Env
            Configures parameters of the Voxelyze environment.

        pop : Population
            This provides the individuals to evaluate.

        print_log : PrintLog()
            For logging with time stamps

        save_vxa_every : int
            Which generations to save information about individual SoftBots

        run_directory : string
            Where to save

        run_name : string
            Experiment name for files

        max_eval_time : int
            How long to run physical simulation per ind in pop

        time_to_try_again : int
            How long to wait until relaunching remaining unevaluated (crashed?) simulations

        save_lineages : bool
            Save the vxa of every ancestor of the surviving individual

        """
        
        # Setting up the simulation object
        self.sim = sim
        self.env = env
        # Setting up the environment object
        if not isinstance(env, list):
            self.env = [env]
        self.save_vxa_every = save_vxa_every
        self.run_directory = run_directory
        self.run_name = run_name
        self.objective_dict = objective_dict
        self.max_gens = max_gens
        self.max_eval_time = max_eval_time
        self.time_to_try_again = time_to_try_again
        self.save_lineages = save_lineages

        
        self.already_evaluated = {}
        self.best_fit_so_far = objective_dict[0]["worst_value"] 

        self.all_evaluated_individuals_ids = []
        self.num_env_cycles = num_env_cycles
        self.curr_env_idx = 0
        self.n_batch = 1
        self.save_nets = save_nets
        # self.save_checkpoint = save_checkpoint
        self.checkpoint_path = os.path.join(self.run_directory, "physics_evaluator_checkpoint.pickle")
        
    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        usePhysicsCache = kwargs["usePhysicsCache"]
        if not resuming_run:
            if usePhysicsCache:
                self.already_evaluated = readFromJson('experiments/physics_evaluator_cache.json')
        initialize_folder_heirarchy(self.run_directory, self.save_nets, save_lineages=self.save_lineages, resuming_run=resuming_run)

    def backup(self, *args, **kwargs):
        saveToPickle(self.checkpoint_path, self)

    def update_env(self):
        if self.num_env_cycles > 0:
            switch_every = self.max_gens / float(self.num_env_cycles)
            self.curr_env_idx += 1 if self.n_batch % switch_every == 0 else 0
            self.curr_env_idx %= len(self.env)
            logger.info(" Using environment {0} of {1}".format(self.curr_env_idx+1, len(self.env)))
    
    def file_recovery(self):
        return readFromPickle(self.checkpoint_path)           

    @timeit
    def evaluate(self, pop, *args, **kwargs):
        self.update_env()        
        self.n_batch += 1
        return pop



class VoxelyzePhysicsEvaluator(BaseSoftBotPhysicsEvaluator):

    def __init__(self, sim, env, save_vxa_every, run_directory, run_name, 
                objective_dict, max_gens, num_env_cycles, max_eval_time=60, 
                time_to_try_again=10, save_lineages=True, save_nets = False, 
                sim_path = '_voxcad', 
                experiments_path = '.'):
        super().__init__(sim, env, save_vxa_every, run_directory, run_name, 
                        objective_dict, max_gens, num_env_cycles, max_eval_time, 
                        time_to_try_again, save_lineages, save_nets)
        self.sim_path = sim_path
        self.experiments_path = experiments_path

    def start(self, **kwargs):
        super().start(**kwargs)
        sub.call(f"cp {self.sim_path}/voxelyzeMain/voxelyze {self.experiments_path}", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine

    @timeit
    def evaluate(self, pop, *args, **kwargs):
        pop_size = kwargs['pop_size']

        if len(pop) == pop_size:
            start_indx = 0
        elif len(pop) > pop_size:
            start_indx = len(pop) - pop_size

        logger.info("Creating folders structure for this generation")
        create_gen_directories(self.n_batch, self.run_directory, self.save_vxa_every, self.save_nets)
        logger.info("Starting voxelyze physics evaluation")
        start_time = time.time()
        num_evaluated_this_gen = 0
        ids_softbot_map = {}

        for ind in pop[start_indx:]:
            # write the phenotype of a SoftBot to a file so that VoxCad can access for self.sim.
            ind.md5 = write_voxelyze_file(self.sim, self.env[self.curr_env_idx], ind, self.run_directory, self.run_name)

            # don't evaluate if invalid
            if not ind.phenotype.is_valid():

                logger.info("Skipping invalid individual")

            # don't evaluate if identical phenotype has already been evaluated
            elif self.env[self.curr_env_idx].actuation_variance == 0 and ind.md5 in self.already_evaluated:
                for rank, goal in self.objective_dict.items():
                    if goal["tag"] is not None:
                        setattr(ind, goal["name"], self.already_evaluated[ind.md5][rank])

                logger.debug("Individual already evaluated:  cached fitness is {}".format(ind.fitness))

                if self.n_batch% self.save_vxa_every == 0 and self.save_vxa_every > 0:
                    sub.call("cp " + self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" % ind.id +
                            " " + self.run_directory + "/Gen_%04i/" % self.n_batch+ self.run_name +
                            "--Gen_%04i--fit_%.08f--id_%05i.vxa" % (self.n_batch, ind.fitness, ind.id), shell=True)

            # otherwise evaluate with voxelyze
            else:

                if ind.id not in ids_softbot_map:
                    num_evaluated_this_gen += 1
                    ids_softbot_map[ind.id] = ind
                    sub.Popen("./voxelyze  -f " + self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" % ind.id,
                            shell=True)

        logger.info("Launched {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, pop_size))

        num_evals_finished = 0
        all_done = False
        already_analyzed_ids = set()
        redo_attempts = 1

        fitness_eval_start_time = time.time()

        def int64Convertion(num):
            if isinstance(num, np.integer):
                return int(num)
            return num

        while not all_done:

            time_waiting_for_fitness = time.time() - fitness_eval_start_time
            # this protects against getting stuck when voxelyze doesn't return a fitness value
            # (diverging self.simulations, crash, error reading .vxa)

            if time_waiting_for_fitness > pop_size * self.max_eval_time:
                # TODO ** WARNING: This could in fact alter the self.sim and undermine the reproducibility **
                all_done = False  # something bad with this individual, probably self.sim diverged
                break

            if time_waiting_for_fitness > pop_size * self.time_to_try_again * redo_attempts:
                # try to redo any self.simulations that crashed
                redo_attempts += 1
                non_analyzed_ids = [idx for idx in ids_softbot_map.keys() if idx not in already_analyzed_ids]
                logger.warning("Rerunning voxelyze for: ", non_analyzed_ids)
                for idx in non_analyzed_ids:
                    sub.Popen("./voxelyze  -f " + self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" % idx,
                            shell=True)

            # check to see if all are finished
            all_done = len(ids_softbot_map) == 0


            f_files_path = self.run_directory + "/fitnessFiles/"
            f_files = [f for f in os.listdir(f_files_path) if os.path.isfile(os.path.join(f_files_path, f))]
            # duplicated ids issue: may be due to entering here two times for the same fitness file found in the directory.

            if not f_files:
                time.sleep(0.5)
                continue

            evaluated_ids = set()
            for f in f_files:
                if "softbotsOutput--id_" in f:
                    evaluated_ids.add((int(f.split("_")[1].split(".")[0]), f))

            if not evaluated_ids:
                time.sleep(0.5)
                continue

            for this_id, file in evaluated_ids:
                ind_filename = os.path.join(f_files_path, file)
                if this_id in already_analyzed_ids:
                    # workaround to avoid any duplicated ids when restarting self.sims
                    logger.info("Duplicate voxelyze results found from THIS gen with id {}".format(this_id))
                    sub.call("rm " + ind_filename, shell=True)

                elif this_id in self.all_evaluated_individuals_ids:
                    logger.info("Duplicate voxelyze results found from PREVIOUS gen with id {}".format(this_id))
                    sub.call("rm " + ind_filename, shell=True)

                else:
                    num_evals_finished += 1
                    already_analyzed_ids.add(this_id)

                    objective_values_dict = read_voxlyze_results(self.objective_dict, logger, ind_filename)

                    logger.info("{0} fit = {1} ({2} / {3})".format(file, objective_values_dict[0],
                                                                        num_evals_finished,
                                                                        num_evaluated_this_gen))

                    # now that we've read the fitness file, we can remove it
                    sub.call("rm " + ind_filename, shell=True)

                    # assign the values to the corresponding individual
                    ind = ids_softbot_map[this_id]
                    #ind = pymoo_ind[0].X
                    for rank, details in self.objective_dict.items():
                        if objective_values_dict[rank] is not None:
                            setattr(ind, details["name"], objective_values_dict[rank])
                        else:
                            for name, details_phenotype in ind.genotype.to_phenotype_mapping.items():
                                if name == details["output_node_name"]:
                                    state = details_phenotype["state"]
                                    setattr(ind, details["name"], details["node_func"](state))

                    self.already_evaluated[ind.md5] = [int64Convertion(getattr(ind, details["name"]))
                                                        for rank, details in
                                                        self.objective_dict.items()]
                    self.all_evaluated_individuals_ids += [this_id]

                    # update the run statistics and file management
                    if ind.fitness > self.best_fit_so_far:
                        self.best_fit_so_far = ind.fitness
                        sub.call("cp " + self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" %
                                ind.id + " " + self.run_directory + "/bestSoFar/fitOnly/" + self.run_name +
                                "--Gen_%04i--fit_%.08f--id_%05i.vxa" %
                                (self.n_batch, ind.fitness, ind.id), shell=True)


                    if self.n_batch% self.save_vxa_every == 0 and self.save_vxa_every > 0:
                        file_source = self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" % ind.id
                        file_destination = self.run_directory + "/Gen_%04i/" % self.n_batch+ self.run_name + "--Gen_%04i--fit_%.08f--id_%05i.vxa" % (self.n_batch, ind.fitness, ind.id)
                        sub.call("mv " + file_source + " " + file_destination, shell=True)
                    else:
                        sub.call("rm " + self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" %
                                ind.id, shell=True)

                    del ids_softbot_map[this_id]


        if not all_done:
            logger.warning("Couldn't get a fitness value in time for some individuals. "
                            "The min fitness was assigned for these individuals")

        logger.info("\nAll Voxelyze evals finished in {} seconds".format(time.time() - start_time))
        logger.info("num_evaluated_this_gen: {0}".format(num_evaluated_this_gen))
        logger.info("Finished voxelyze physics evaluation")

        return super().evaluate(pop)



class VoxcraftPhysicsEvaluator(BaseSoftBotPhysicsEvaluator):

    def __init__(self, sim, env, save_vxa_every, run_directory, run_name, 
                objective_dict, max_gens, num_env_cycles, max_eval_time=60, 
                time_to_try_again=10, save_lineages=True, save_nets = False, 
                sim_path = '_voxcraft-sim', 
                experiments_path = 'experiments'):
        super().__init__(sim, env, save_vxa_every, run_directory, run_name, 
                        objective_dict, max_gens, num_env_cycles, max_eval_time, 
                        time_to_try_again, save_lineages, save_nets)
        self.sim_path = sim_path
        self.experiments_path = experiments_path

    def start(self, **kwargs):
        super().start(**kwargs)
        sub.call(f"cp {self.sim_path}/build/voxcraft-sim .", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine
        sub.call(f"cp {self.sim_path}/build/vx3_node_worker .", shell=True)
        sub.call(f"cp {self.sim_path}/demos/voxelyze/base.vxa {self.run_directory}/voxelyzeFiles/", shell=True)

    @timeit
    def evaluate(self, pop, *args, **kwargs):
        pop_size = kwargs['pop_size']

        if len(pop) == pop_size:
            start_indx = 0
        elif len(pop) > pop_size:
            start_indx = len(pop) - pop_size

        start_time = time.time()
        num_evaluated_this_gen = 0
        # ids_to_analyze = []
        ids_softbot_map = {}
        logger.info("Creating folders structure for this generation")
        create_gen_directories(self.n_batch, self.run_directory, self.save_vxa_every, self.save_nets)
        logger.info("Starting voxcraft physics evaluation")

            
        for ind in pop[start_indx:]:
            # write the phenotype of a SoftBot to a file so that VoxCad can access for self.sim.
            ind.md5, root = get_vxd(self.sim, self.env[self.curr_env_idx], ind)
            write_voxelyze_file(self.sim, self.env[self.curr_env_idx], ind, self.run_directory, self.run_name)
            # don't evaluate if invalid
            if not ind.phenotype.is_valid():
                logger.info("Skipping invalid individual")

            # don't evaluate if identical phenotype has already been evaluated
            elif self.env[self.curr_env_idx].actuation_variance == 0 and ind.md5 in self.already_evaluated:
                
                for rank, goal in self.objective_dict.items():
                    if goal["tag"] is not None:
                        if goal["name"] == "fitness":
                            logger.info(f"Individual with id->{ind.id} and hash->{ind.md5}... has already been evaluated with fitness->{self.already_evaluated[ind.md5][rank]}")
                        setattr(ind, goal["name"], self.already_evaluated[ind.md5][rank])

                if self.n_batch% self.save_vxa_every == 0 and self.save_vxa_every > 0:
                    source_file = self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" % ind.id 
                    dest_file = self.run_directory + "/Gen_%04i/" % self.n_batch+ self.run_name + "--Gen_%04i--fit_%.08f--id_%05i--md5_%s.vxa" % (self.n_batch, ind.fitness, ind.id, ind.md5)
                    if os.path.exists(source_file):
                        sub.call("cp " + source_file +
                                " " + dest_file, shell=True)

            # otherwise evaluate with voxcraft
            else:

                if ind.id not in ids_softbot_map:
                    num_evaluated_this_gen += 1
                    ids_softbot_map[ind.id] = ind
                    with open(self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxd" % ind.id, "w", encoding='utf-8') as vxd:
                        root_str = etree.tostring(root)
                        vxd.write(root_str.decode('utf-8'))

        all_done = len(ids_softbot_map) == 0
        fitness_eval_start_time = time.time()



        while not all_done:
            time_waiting_for_fitness = time.time() - fitness_eval_start_time
            if time_waiting_for_fitness > pop_size * self.max_eval_time:
                break
            try:
                process = sub.run(f"./voxcraft-sim -f -i {self.run_directory}/voxelyzeFiles -o {self.run_directory}/output.xml", universal_newlines=True, stdout=sub.PIPE, stderr=sub.PIPE, shell=True)
                
                logger.info(process.stdout)
                logger.info(process.stderr)
                # logger.info(process.returncode)
                
                if "CUDA Function Error: out of memory" in process.stdout:
                    raise MemoryError("\033[91mvoxcraft-sim/src/VX3/VX3_SimulationManager.cu(425): CUDA Function Error: out of memory\033[0m")
                if "CUDA Function Error: too many resources requested for launch" in process.stdout:
                    raise ResourceWarning("\033[91mvoxcraft-sim/src/VX3/VX3_SimulationManager.cu(425): CUDA Function Error: too many resources requested for launch\033[0m")
                # sub.run waits for the process to return
                # after it does, we collect the results output by the simulator
                fitness_report = etree.parse(f"{self.run_directory}/output.xml").getroot()
                all_done = True

            except IOError as io:
                logger.error(f"There was an IOError:")
                logger.exception(io)
                logger.error(f"Re-simulating this batch again...")

            except IndexError as ie:
                logger.error(f"There was an IndexError:")
                logger.exception(ie)
                logger.error(f"Re-simulating this batch again...")
            except MemoryError as me:
                logger.error(f"There was Memory error:")
                logger.exception(me)
                logger.error(f"Re-simulating this batch again...")
            except ResourceWarning as re:
                logger.error(f"There was resource usage error:")
                logger.exception(re)
                logger.error(f"Re-simulating this batch again...")

        def int64Convertion(num):
            if isinstance(num, np.integer):
                return int(num)
            return num

        for ind_id, ind in ids_softbot_map.items():
            
            results = {rank: None for rank in range(len(self.objective_dict))}
            for rank, details in self.objective_dict.items():
                tag = details["tag"]
                if tag is not None:
                    tag = tag.lstrip('<').rstrip('>')
                    tag_ocurrences = fitness_report.findall("./detail/" + self.run_name + "--id_%05i" % ind_id + "/" + tag)
                    results[rank] = float(tag_ocurrences[0].text)

            for rank, details in self.objective_dict.items():
                if results[rank] is not None:
                    if details["name"] == "fitness":
                        logger.info(f"Individual with id->{ind.id} and hash->{ind.md5} was evaluated with fitness->{results[rank]}")
                    setattr(ind, details["name"], results[rank])
                else:
                    for name, details_phenotype in ind.genotype.to_phenotype_mapping.items():
                        if name == details["output_node_name"]:
                            state = details_phenotype["state"]
                            setattr(ind, details["name"], details["node_func"](state))



            self.already_evaluated[ind.md5] = [int64Convertion(getattr(ind, details["name"]))
                                                for rank, details in
                                                self.objective_dict.items()]


        # for ind in pop[start_indx:]:
        #     ind_id = ind.id
            ind_filename_vxd = self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxd" % ind_id
            ind_filename_vxa = self.run_directory + "/voxelyzeFiles/" + self.run_name + "--id_%05i.vxa" % ind_id
            if os.path.exists(ind_filename_vxd):
                sub.call("rm " + ind_filename_vxd, shell=True)

            # update the run statistics and file management
            if ind.fitness > self.best_fit_so_far:
                self.best_fit_so_far = ind.fitness
                file_destination = self.run_directory + "/bestSoFar/fitOnly/" + self.run_name + "--Gen_%04i--fit_%.08f--id_%05i--md5_%s.vxa" % (self.n_batch, ind.fitness, ind_id, ind.md5)

                if os.path.exists(ind_filename_vxa):
                    sub.call("cp " + ind_filename_vxa + " " + file_destination, shell=True)


            if self.n_batch% self.save_vxa_every == 0 and self.save_vxa_every > 0:
                if os.path.exists(ind_filename_vxa):
                    file_destination = self.run_directory + "/Gen_%04i/" % self.n_batch+ self.run_name + "--Gen_%04i--fit_%.08f--id_%05i--md5_%s.vxa" % (self.n_batch, ind.fitness, ind_id, ind.md5)
                    sub.call("mv " + ind_filename_vxa + " " + file_destination, shell=True)
            else:
                sub.call("rm " + ind_filename_vxa, shell=True)

        if not all_done:
            logger.warning("Couldn't get a fitness value in time for some individuals. "
                            "The min fitness was assigned for these individuals")

        logger.info("All voxcraft evals finished in {} seconds".format(time.time() - start_time))
        logger.info("num_evaluated_this_gen: {0}".format(num_evaluated_this_gen))

        logger.info("Finished voxcraft physics evaluation")

        return super().evaluate(pop)