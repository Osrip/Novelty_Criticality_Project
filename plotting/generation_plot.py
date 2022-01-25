import matplotlib.pyplot as plt
import numpy as np
from helper_functions.loading_saving import load_all_files_that_include_as_dict
from helper_functions.loading_saving import load_pickle_compressed
from helper_functions.plotting import random_color
from helper_functions.loading_saving import mkdir


def main(plot_settings):
    '''
    Plot fitness for generations during evolution
    '''
    gen_dict = load_data(plot_settings['sim_name'], plot_settings)
    plot_species_scatter(gen_dict, plot_settings)


def load_data(sim_name, plot_settings):
    '''
    Load all required data
    '''
    save_heat_cap_path = 'save/{}/generations/'.format(sim_name)
    include_name = 'generation'

    # save_heat_name = 'heat_cap_ind_{}.pickle'.format(genome.key)
    gen_dict = load_all_files_that_include_as_dict(include_name, save_heat_cap_path)

    return gen_dict


def get_species_num(gen_dict):
    '''
    Returns number of species in simulation
    '''

    max_species = []
    for gen, pop_dict in gen_dict.items():
        max_species.append(np.max([species for genome, species in pop_dict['species_dict']['genome_to_species'].items()]))
    return np.max(max_species)



def plot_species_scatter(gen_dict, plot_settings):
    gen_list = []
    pop_dict_list = []
    for gen, pop_dict in gen_dict.items():
        gen_list.append(gen)
        pop_dict_list.append(pop_dict)
    pop_dict_list = np.array(pop_dict_list)
    pop_dict_list_sorted = list(pop_dict_list[np.argsort(gen_list)])
    species_num = get_species_num(gen_dict)
    species_to_color = [random_color() for _ in range(species_num+1)]


    x_generation = []
    y_attr = []
    c_species = []

    x_gen_mean = []
    y_attr_mean = []
    c_species_mean = []

    for pop_dict in pop_dict_list_sorted:
        genome_to_species = pop_dict['species_dict']['genome_to_species']
        species_to_genomes = {v: k for k, v in genome_to_species.items()}
        for ind_num, ind in pop_dict['population'].items():
            attr_val = getattr(ind, plot_settings['attr'])
            x_generation.append(pop_dict['generation'])
            y_attr.append(attr_val)
            species_num = genome_to_species[ind_num]
            c_species.append(species_to_color[species_num])
    plt.scatter(x_generation, y_attr, c=c_species, s=3, alpha=0.35)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    savedir = 'save/{}/figs/generation_plot/'.format(plot_settings['sim_name'])
    savename = 'gen_plot.png'
    mkdir(savedir)
    plt.savefig(savedir + savename, dpi=300)
    print('Saving plot: {}{}'.format(savedir, savename))
    plt.show()




def pop_dict_to_list_of_attr(pop_dict, attr):
    '''
    returns sorted list of attributes
    '''
    ind_nums = []

    for ind_num, ind_obj in pop_dict:
        getattr(ind_obj, attr)

if __name__ == '__main__':
    '''
    PLOT SETTINGS
    The created plots show the evolution of a desired genome attribute (for example fitness) 
    with respect to different species. Generations on x-axis, Attribute on y-axis
    Practical set-up for playing around...
    The figures are saved in each according simulation folder
    '''

    plot_settings = {}
    # List all folder names of simulation runs, that you wish to create plots for:
    sim_names = ['sim_20220125-133002_Symmetric_calc_heat_cap_many_generations_with_heat_caps']
    for sim_name in sim_names:
        print('Loading sim {}'.format(sim_name))
        plot_settings['sim_name'] = sim_name
        # Specify the attribute, you would like to plot (any attribute of the genome class specified in a string)
        plot_settings['attr'] = 'fitness'
        main(plot_settings)