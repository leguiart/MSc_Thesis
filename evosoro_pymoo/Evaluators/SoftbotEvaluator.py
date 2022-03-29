import os
import sys
import time
import random
import numpy as np
import subprocess as sub
#sys.path.append(os.getcwd() + "/../..")
from evosoro.tools.read_write_voxelyze import read_voxlyze_results, write_voxelyze_file

def evaluate_all_pymoo(sim, env, pop, print_log, save_vxa_every, run_directory, run_name, already_evaluated, all_evaluated_individuals_ids, objective_dict, problem, max_eval_time=60,
                time_to_try_again=10, save_lineages = True):
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
    start_time = time.time()
    num_evaluated_this_gen = 0
    ids_to_analyze = []

    for pymoo_ind in pop:
        ind = pymoo_ind[0].X
        # write the phenotype of a SoftBot to a file so that VoxCad can access for sim.
        ind.md5 = write_voxelyze_file(sim, env, ind, run_directory, run_name)

        # don't evaluate if invalid
        if not ind.phenotype.is_valid():
            for rank, goal in objective_dict.items():
                if goal["name"] != "age":
                    setattr(ind, goal["name"], goal["worst_value"])
            print_log.message("Skipping invalid individual")

        # don't evaluate if identical phenotype has already been evaluated
        elif env.actuation_variance == 0 and ind.md5 in already_evaluated:
            for rank, goal in objective_dict.items():
                if goal["tag"] is not None:
                    setattr(ind, goal["name"], already_evaluated[ind.md5][rank])
            print_log.message("Individual already evaluated:  cached fitness is {}".format(ind.fitness))

            if problem.n_gen % save_vxa_every == 0 and save_vxa_every > 0:
                sub.call("cp " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" % ind.id +
                         " " + run_directory + "/Gen_%04i/" % problem.n_gen + run_name +
                         "--Gen_%04i--fit_%.08f--id_%05i.vxa" % (problem.n_gen, ind.fitness, ind.id), shell=True)

        # otherwise evaluate with voxelyze
        else:
            num_evaluated_this_gen += 1
            # pop.total_evaluations += 1
            ids_to_analyze += [ind.id]

            sub.Popen("./voxelyze  -f " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" % ind.id,
                      shell=True)

    print_log.message("Launched {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop)))

    num_evals_finished = 0
    all_done = False
    already_analyzed_ids = []
    redo_attempts = 1

    fitness_eval_start_time = time.time()

    while not all_done:

        time_waiting_for_fitness = time.time() - fitness_eval_start_time
        # this protects against getting stuck when voxelyze doesn't return a fitness value
        # (diverging simulations, crash, error reading .vxa)

        if time_waiting_for_fitness > len(pop) * max_eval_time:
            # TODO ** WARNING: This could in fact alter the sim and undermine the reproducibility **
            all_done = False  # something bad with this individual, probably sim diverged
            break

        if time_waiting_for_fitness > len(pop) * time_to_try_again * redo_attempts:
            # try to redo any simulations that crashed
            redo_attempts += 1
            non_analyzed_ids = [idx for idx in ids_to_analyze if idx not in already_analyzed_ids]
            print ("Rerunning voxelyze for: ", non_analyzed_ids)
            for idx in non_analyzed_ids:
                sub.Popen("./voxelyze  -f " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" % idx,
                          shell=True)

        # check to see if all are finished
        all_done = True
        for pymoo_ind in pop:
            ind = pymoo_ind[0].X
            if ind.phenotype.is_valid() and ind.fitness == objective_dict[0]["worst_value"]:
                all_done = False

        # check for any fitness files that are present
        ls_check = sub.check_output(["ls", run_directory + "/fitnessFiles/"])
        # duplicated ids issue: may be due to entering here two times for the same fitness file found in the directory.

        if ls_check:
            # ls_check = random.choice(ls_check.split())  # doesn't accomplish anything and undermines reproducibility
            ls_check = ls_check.split()[0].decode('ascii')
            if "softbotsOutput--id_" in ls_check:
                this_id = int(ls_check.split("_")[1].split(".")[0])

                if this_id in already_analyzed_ids:
                    # workaround to avoid any duplicated ids when restarting sims
                    print_log.message("Duplicate voxelyze results found from THIS gen with id {}".format(this_id))
                    sub.call("rm " + run_directory + "/fitnessFiles/" + ls_check, shell=True)

                elif this_id in all_evaluated_individuals_ids:
                    print_log.message("Duplicate voxelyze results found from PREVIOUS gen with id {}".format(this_id))
                    sub.call("rm " + run_directory + "/fitnessFiles/" + ls_check, shell=True)

                else:
                    num_evals_finished += 1
                    already_analyzed_ids.append(this_id)

                    ind_filename = run_directory + "/fitnessFiles/" + ls_check
                    objective_values_dict = read_voxlyze_results(objective_dict, print_log, ind_filename)

                    print_log.message("{0} fit = {1} ({2} / {3})".format(ls_check, objective_values_dict[0],
                                                                         num_evals_finished,
                                                                         num_evaluated_this_gen))

                    # now that we've read the fitness file, we can remove it
                    sub.call("rm " + run_directory + "/fitnessFiles/" + ls_check, shell=True)

                    # assign the values to the corresponding individual
                    for pymoo_ind in pop:
                        ind = pymoo_ind[0].X
                        if ind.id == this_id:
                            for rank, details in objective_dict.items():
                                if objective_values_dict[rank] is not None:
                                    setattr(ind, details["name"], objective_values_dict[rank])
                                else:
                                    for name, details_phenotype in ind.genotype.to_phenotype_mapping.items():
                                        if name == details["output_node_name"]:
                                            state = details_phenotype["state"]
                                            setattr(ind, details["name"], details["node_func"](state))

                            already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                                              for rank, details in
                                                              objective_dict.items()]
                            all_evaluated_individuals_ids += [this_id]

                            # update the run statistics and file management
                            if ind.fitness > problem.best_fit_so_far:
                                problem.best_fit_so_far = ind.fitness
                                sub.call("cp " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" %
                                         ind.id + " " + run_directory + "/bestSoFar/fitOnly/" + run_name +
                                         "--Gen_%04i--fit_%.08f--id_%05i.vxa" %
                                         (problem.n_gen, ind.fitness, ind.id), shell=True)

                            # if save_lineages:
                            #     sub.call("cp " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" %
                            #              ind.id + " " + run_directory + "/ancestors/", shell=True)

                            if problem.n_gen % save_vxa_every == 0 and save_vxa_every > 0:
                                file_source = run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" % ind.id
                                file_destination = run_directory + "/Gen_%04i/" % problem.n_gen + run_name + "--Gen_%04i--fit_%.08f--id_%05i.vxa" % (problem.n_gen, ind.fitness, ind.id)
                                sub.call("mv " + file_source + " " + file_destination, shell=True)
                            else:
                                sub.call("rm " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" %
                                         ind.id, shell=True)

                            break

            # wait a second and try again
            else:
                time.sleep(0.5)
        else:
            time.sleep(0.5)

    if not all_done:
        print_log.message("WARNING: Couldn't get a fitness value in time for some individuals. "
                          "The min fitness was assigned for these individuals")

    print_log.message("\nAll Voxelyze evals finished in {} seconds".format(time.time() - start_time))
    print_log.message("num_evaluated_this_gen: {0}".format(num_evaluated_this_gen))
    # print_log.message("total_evaluations: {}".format(pop.total_evaluations))