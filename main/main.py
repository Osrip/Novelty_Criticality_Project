
import os
import ray
import click
import gym
# import neat
import neat_local as neat
import time
import torch

from modules.novelty_env_eval_save_inputs import MultiEnvEvaluatorNovelty
from modules.multi_env_eval_save_inputs import MultiEnvEvaluator
from helper_functions.novelty_search import novelty_population

from modules.neat_reporter import LogReporter
# CHOOSE AGENT NETWORK:
from modules.ising_net import IsingNet as AgentNet
# from modules.recurrent_net import RecurrentNet as AgentNet
from modules.save_file_reporter import SaveReporter
from helper_functions.loading_saving import save_settings
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")


def make_env():
    '''
    Create an Open-AI Gym environment
    '''
    return gym.make("CartPole-v0")


def make_net(genome, config, global_dict, bs):
    '''
    Create agent networks from genomes (Ising-based and Recurrent networks are possible)
    '''
    return AgentNet.create(genome, config, global_dict, bs)


def activate_net(net, states):
    '''
    Activate Networks
    '''
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5


def make_save_dir(global_dict):
    '''
    Create directory, that simulation is saved in
    '''
    if global_dict['append_sim_name'] == '':
        save_dir = 'save/sim_{}/'.format(time.strftime("%Y%m%d-%H%M%S"))
    else:
        save_dir = 'save/sim_{}_{}/'.format(time.strftime("%Y%m%d-%H%M%S"), global_dict['append_sim_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def run(global_dict):
    '''
    Main function, that assembles all modules together and runs the simulation
    '''
    torch.set_num_threads(global_dict['cores'])

    n_generations = global_dict['n_generations']
    global_dict['save_dir'] = make_save_dir(global_dict)

    if global_dict['device'] == 'cuda':
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        global_dict['device'] = dev #torch.device(dev)

    save_settings(global_dict)
    copyfile(global_dict['config_file'], '{}/settings/{}'.format(global_dict['save_dir'], global_dict['config_file']))

    # Load the config file, which is assumed to live in the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), global_dict['config_file'])
    # config_path = os.path.join(os.path.dirname(__file__), "neat_playaround_doubling_size.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    #  Create the evaluator, where agent evaluation is done. Compatible with all gy environments.
    # This evaluator is exclusively for traditional objective-based fitness functions
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, global_dict, make_env=make_env, max_env_steps=global_dict['max_env_steps'],
        batch_size=global_dict['batch_size']
    )

    #  Create the evaluator, where agent evaluation is done. Compatible with all gy environments.
    # This evaluator is exclusively for novelty-based fitness functions
    evaluator_novelty = MultiEnvEvaluatorNovelty(
        make_net, activate_net, global_dict, make_env=make_env, max_env_steps=global_dict['max_env_steps'],
        batch_size=global_dict['batch_size']
    )


    def eval_genomes(genomes, config, global_dict, display = False):
        '''
        Evaluates all genomes in the given gym environment using a traditional objective-based fitness function
        :param display: If activated, the evaluation of one individual is shown as an animation
        '''
        for i, (_, genome) in enumerate(genomes):
            # This line displays the first individual of each generation live
            display_env = i == 0 and display
            genome.fitness = evaluator.eval_genome(genome, config, global_dict, display_env=display_env)


    def eval_genomes_novelty(genomes, config, global_dict, display=False):
        '''
        Evaluates all genomes in the given gym environment using a novelty-based fitness function
        :param display: If activated, the evaluation of one individual is shown as an animation
        '''
        all_states_mats = []
        for i, (_, genome) in enumerate(genomes):
            # This line displays the first individual of each generation live
            display_env = i == 0 and display
            genome.performance, all_states_mat = evaluator_novelty.eval_genome(genome, config, global_dict, display_env=display_env)
            all_states_mats.append(all_states_mat)

        avg_genome_behav_dists = novelty_population(all_states_mats, genomes, global_dict)
        for (_, genome), behav_dist in zip(genomes, avg_genome_behav_dists):
            genome.novelty = behav_dist
            genome.fitness = behav_dist
                # genome.fitness


    def eval_genomes_parallel(genomes, config, global_dict, display=False):
        '''
        Evaluates all genomes in the given gym environment using a traditional objective-based fitness function
        Caluclations are done in parallel using ray
        :param display: If activated, the evaluation of one individual is shown as an animation
        '''
        ray.init()
        ray_funcs = [evaluate_parallel_helper.remote(
            genome, config, global_dict, display_env=i == 0 and display) for i, (_, genome) in enumerate(genomes)]
        fitnesses = ray.get(ray_funcs)
        for fitness, (_, genome) in zip(fitnesses, genomes):
            genome.fitness = fitness
        ray.shutdown()


    @ray.remote
    def evaluate_parallel_helper(genome, config, global_dict, display_env=False):
        '''
        Helper function of eval_genomes_parallel called by Ray.
        '''
        fitness = evaluator.eval_genome(genome, config, global_dict, display_env=display_env)
        return fitness

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    # reporter = neat.StdOutReporter(True)
    reporter = SaveReporter(global_dict)
    pop.add_reporter(reporter)
    #  This evaluates the best genome of the generation!!!
    logger = LogReporter("neat.log", evaluator.eval_genome, global_dict)
    pop.add_reporter(logger)

    if global_dict['novelty_search']:
        pop.run(eval_genomes_novelty, global_dict, n_generations)
    else:
        pop.run(eval_genomes, global_dict, n_generations)


if __name__ == "__main__":
    # Practical set-up to play around with the settings:
    global_dict = {}
    # -------------------
    # EXPERIMENT SETTINGS:
    # -------------------

    # Hyper parameters for simulation are loaded from the config file
    global_dict['config_file'] = "neat_playaround.cfg"
    # Device that is used for the PyTorch calculations (Either GPU-->'cuda' or CPU--> 'cpu')
    global_dict['device'] = 'cpu'  # either 'cpu' or 'cuda'
    # Make Connectivity of the network symmetric (J_ij = J_ji)
    global_dict['make_edges_symmetric'] = True

    # Number of maximal generations in the genetic algorithm NEAT
    global_dict['n_generations'] = 101
    # Number of maxmimal time steps in gym enviropnment ('None' is also an option)
    global_dict['max_env_steps'] = None
    # Batch size gives number of environments that a genome / an agent is evaluated in simultaneously
    # Implemented as extra dimension in matriy multiplications in PyTorch, especially efficient on GPU
    global_dict['batch_size'] = 2

    # Printing out extra information about SPECIATION in NEAT each generation:
    global_dict['show_species_detail'] = False

    # HEAT CAPACITY  CALCULATIONS
    # Hyper parameters fpor heat capacity calculations:
    global_dict['n_internal_heat_cap'] = 1 #10
    global_dict['exp_c_beta_min'] = -2
    global_dict['exp_c_beta_max'] = 2
    global_dict['num_c_beta'] = 10 #100  # 100
    global_dict['num_batches_heat_cap'] = 10 #1000
    global_dict['internal_steps_before_recording_heat_cap'] = 0

    # Heat capacity calculations have to be initialized at energy minimum.
    # Find energy minimum brute force (inefficient for large networks)
    # If False simulated annealing heuristic is used
    global_dict['brute_force_init'] = False #  If false initialize use simulated annealing heuristic for init

    # Perform Heat capacity calculations: (Can be plotted with plotting/heat_cap_plot.py)
    global_dict['calc_heat_caps'] = True
    # Perform heat capacity calculation every Nth generation
    global_dict['heat_cap_nth_gen'] = 20

    # Calculate susceptibility: (Can be plotted with plotting/heat_cap_plot.py)
    global_dict['calc_susceptibility'] = False
    global_dict['susceptibility_nth_gen'] = 5
    global_dict['include_sensors_in_susceptibility'] = False

    # Save the network enery values during the calculation of the heat capacity and susceptibility
    # (Can be plotted with plotting/energy_during_thermalization_plot.py)
    global_dict['save_thermalization'] = False
    global_dict['save_thermalization_gens'] = [0]

    # Simulated annealing for initializing heat capacity calculations
    global_dict['n_internal_anneal'] = 1000  # 100
    global_dict['exp_beta_min_anneal'] = -2
    global_dict['exp_beta_max_anneal'] = 5
    # TODO: For higher temperatures it may be good not to initialize the lowest possible energy as in
    # "praxis" this low energy state is never experienced by system
    # Suggestion: only thermalize to temperature, that system actually has for c_beta = 0

    # Magnetization (Can be plotted with plotting/heat_cap_plot.py)
    # 'include_sensors_in_susceptibility' and all other parameters from heat capacity
    # (except n_internal_steps) apply
    global_dict['calc_magnetization'] = False
    global_dict['magnetization_nth_gen'] = 50
    global_dict['repeats_per_batch_magnetization'] = 10

    # Number of cores for PyTorch parallelization on CPU
    global_dict['cores'] = 16

    # Folder name for current experminet
    global_dict['append_sim_name'] = 'Symmetric_calc_heat_cap'

    # NOVELTY SEARCH: Uses a novelty base-fitness function
    # If novelty_search == False an objective-based fitness function is used
    global_dict['novelty_search'] = True
    global_dict['bin_size_compare_behav'] = 10

    run(global_dict)
