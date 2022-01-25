
import os

import numpy as np
import time
from modules.ising_net import IsingNet
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
from helper_functions.loading_saving import save_pickle_compressed
from helper_functions.loading_saving import mkdir
import copy

class SaveReporter(object):
    '''
    A Reporter object is called by the NEAT-Python package at different points during the evolutaion process in the
    genetic algorithm.

    The SaveReporter class prints out at saves desired information at those points during the evaluation process
    and also executes additional processing of the data (such as calculating the heat capacity)
     prior to saving

    Methods
    ---
    start_generation(self, generation)

    end_generation(self, config, population, species_set)

    post_evaluate(self, config, population, species, best_genome)
        All actions done post evaluation
        Here the heat capacity / susceptibility / magnetization is calculated if required for the current generation

    post_reproduction(self, config, population, species)

    post_reproduction(self, config, population, species)

    found_solution(self, config, generation, best)
        Executed when fitness threshold is met

    species_stagnant(self, sid, species)

    info(self, msg)
    '''

    def __init__(self, global_dict):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.save_dir = global_dict['save_dir']
        self.global_dict = global_dict

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.global_dict['show_species_detail']:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))



    def post_evaluate(self, config, population, species, best_genome):
        '''
        All actions done post evaluation
        Here the heat capacity / susceptibility / magnetization is calculated if required for the current generation
        '''
        # HEAT CAPACITY IS CALCULATED BEFORE REPRODUCTION
        # Calculate heat capacity
        for key, genome in population.items():
            if genome.generation % self.global_dict['heat_cap_nth_gen'] == 0 and self.global_dict['calc_heat_caps']:
                calc_heat_cap = True
            else:
                calc_heat_cap = False

            if genome.generation % self.global_dict['susceptibility_nth_gen'] == 0 and self.global_dict['calc_susceptibility']:
                calc_susceptibility = True
            else:
                calc_susceptibility = False

            if genome.generation % self.global_dict['magnetization_nth_gen'] == 0 and self.global_dict['calc_magnetization']:
                calc_magnetization = True
            else:
                calc_magnetization = False

            break
        if calc_heat_cap:
            print('Calculating heat capacity')
            for key, genome in population.items():
                net = IsingNet.create(genome, config, self.global_dict)
                c_betas, heat_caps_c_beta = net.calculate_heat_capacity(self.global_dict['n_internal_heat_cap'],
                                                                        self.global_dict['exp_c_beta_min'],
                                                                        self.global_dict['exp_c_beta_max'],
                                                                        self.global_dict['num_c_beta'])
                # TODO I deleted this line
                # genome.heat_caps = heat_caps_c_beta

                save_heat_cap_path = '{}/heat_cap_data/gen{}'.format(self.global_dict['save_dir'], genome.generation)
                save_heat_name = 'heat_cap_ind_{}.pickle'.format(genome.key)
                mkdir(save_heat_cap_path)
                save_pickle_compressed('{}/{}'.format(save_heat_cap_path, save_heat_name), heat_caps_c_beta)

            save_beta_name = 'c_betas.pickle'
            save_pickle_compressed('{}/{}'.format(save_heat_cap_path, save_beta_name), c_betas)

        if calc_susceptibility:
            print('Calculating magnetic susceptibility')
            for key, genome in population.items():
                net = IsingNet.create(genome, config, self.global_dict)
                c_betas, heat_caps_c_beta = net.calculate_heat_capacity(self.global_dict['n_internal_heat_cap'],
                                                                        self.global_dict['exp_c_beta_min'],
                                                                        self.global_dict['exp_c_beta_max'],
                                                                        self.global_dict['num_c_beta'],
                                                                        magnetic_susceptibility=True)
                # TODO I deleted this line
                # genome.heat_caps = heat_caps_c_beta

                save_heat_cap_path = '{}/susceptibility_data/gen{}'.format(self.global_dict['save_dir'], genome.generation)
                save_heat_name = 'susceptibility_ind_{}.pickle'.format(genome.key)
                mkdir(save_heat_cap_path)
                save_pickle_compressed('{}/{}'.format(save_heat_cap_path, save_heat_name), heat_caps_c_beta)

            save_beta_name = 'c_betas.pickle'
            save_pickle_compressed('{}/{}'.format(save_heat_cap_path, save_beta_name), c_betas)

        if calc_magnetization:
            print('Calculating magnetic magnetization')
            for key, genome in population.items():
                net = IsingNet.create(genome, config, self.global_dict)
                c_betas, heat_caps_c_beta = net.calculate_heat_capacity(self.global_dict['repeats_per_batch_magnetization'],
                                                                        self.global_dict['exp_c_beta_min'],
                                                                        self.global_dict['exp_c_beta_max'],
                                                                        self.global_dict['num_c_beta'],
                                                                        magnetization=True)
                # TODO I deleted this line
                # genome.heat_caps = heat_caps_c_beta

                save_heat_cap_path = '{}/magnetization_data/gen{}'.format(self.global_dict['save_dir'], genome.generation)
                save_heat_name = 'magnetization_ind_{}.pickle'.format(genome.key)
                mkdir(save_heat_cap_path)
                save_pickle_compressed('{}/{}'.format(save_heat_cap_path, save_heat_name), heat_caps_c_beta)

            save_beta_name = 'c_betas.pickle'
            save_pickle_compressed('{}/{}'.format(save_heat_cap_path, save_beta_name), c_betas)

        # Save all data, that is later required (quick loading files contain less information but can be loaded
        # faster due to reduced size)

        # Save pickle.pgz
        # genome_dicts = [genome_attr_to_dict(genome) for genome in population]
        species_dict = species_set_attr_to_dict(species)
        save_dict = {'generation': self.generation, 'population': population, 'species_dict': species_dict}
        if not os.path.exists('{}/generations'.format(self.global_dict['save_dir'])):
            os.makedirs('{}/generations'.format(self.global_dict['save_dir']))
        pickle_name = '{}/generations/generation{}.pickle'.format(self.global_dict['save_dir'], str(self.generation).zfill(6))
        save_pickle_compressed(pickle_name, save_dict)

        # Save quick loading file
        fitnesses = [c.fitness for c in itervalues(population)]
        genome_to_species = species.genome_to_species
        save_dict_quick = {'generation': self.generation, 'fitnesses': fitnesses, 'genome_to_species': genome_to_species}
        if not os.path.exists('{}/generations_quick'.format(self.global_dict['save_dir'])):
            os.makedirs('{}/generations_quick'.format(self.global_dict['save_dir']))
        pickle_name_quick = '{}/generations_quick/generation_quick{}.pickle'.format(self.global_dict['save_dir'], str(self.generation).zfill(6))
        save_pickle_compressed(pickle_name_quick, save_dict_quick)

        # Print interesting information during the run

        # Avg fitness + best genome
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def post_reproduction(self, config, population, species):
        pass

    def post_reproduction(self, config, population, species):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        pass
        # print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)


def species_set_attr_to_dict(species_set):
    attr_dict = {}
    attr_dict['genome_to_species'] = species_set.genome_to_species
    attr_dict['species'] = species_set.species
    return attr_dict


def genome_attr_to_dict(genome):
    attr_dict = {}
    attr_dict['fitness'] = genome.fitness
    attr_dict['key'] = genome.key
    return attr_dict
