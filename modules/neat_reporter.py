
import json
import time
from pprint import pprint

import numpy as np
from neat.reporting import BaseReporter


class LogReporter(BaseReporter):
    '''
    A Reporter object is called by the NEAT-Python package at different points during the evolutaion process in the
    genetic algorithm.

    The LogReporter class only prints minimal information about average performance and the best performing individual

    Methods
    ---
    start_generation(self, generation)

    end_generation(self, config, population, species_set)

    post_evaluate(self, config, population, species, best_genome)

    complete_extinction(self)

    found_solution(self, config, generation, best)

    species_stagnant(self, sid, species)
    '''
    def __init__(self, fnm, eval_best, global_dict, eval_with_debug=False):
        self.log = open(fnm, "a")
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.eval_best = eval_best
        self.eval_with_debug = eval_with_debug
        self.log_dict = {}

        self.global_dict = global_dict

    def start_generation(self, generation):
        self.log_dict["generation"] = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        self.log_dict["pop_size"] = ng

        ns = len(species_set.species)
        self.log_dict["n_species"] = ns

        elapsed = time.time() - self.generation_start_time
        self.log_dict["time_elapsed"] = elapsed

        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = np.mean(self.generation_times)
        self.log_dict["time_elapsed_avg"] = average

        self.log_dict["n_extinctions"] = self.num_extinctions

        pprint(self.log_dict)
        self.log.write(json.dumps(self.log_dict) + "\n")

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = np.mean(fitnesses)
        fit_std = np.std(fitnesses)

        self.log_dict["fitness_avg"] = fit_mean
        self.log_dict["fitness_std"] = fit_std

        self.log_dict["fitness_best"] = best_genome.fitness

        print("=" * 50 + " Best Genome: " + "=" * 50)
        if self.eval_with_debug:
            print(best_genome)

        best_fitness_val = self.eval_best(
            best_genome, config, self.global_dict, debug=self.eval_with_debug, suppress_input_saving=True
        )
        self.log_dict["fitness_best_val"] = best_fitness_val

        n_neurons_best, n_conns_best = best_genome.size()
        self.log_dict["n_neurons_best"] = n_neurons_best
        self.log_dict["n_conns_best"] = n_conns_best

    def complete_extinction(self):
        self.num_extinctions += 1

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass
