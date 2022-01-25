import matplotlib.pyplot as plt
import numpy as np

from helper_functions.loading_saving import load_files_ints_after_substr
from helper_functions.loading_saving import mkdir
from helper_functions.loading_saving import load_all_files_that_include


def main_thermalization(plot_settings):
    '''
    Visualize the change in the network energy during the metropolis-hastings equilibrization
    '''
    plot_data = load_files_thermalize('energies_thermalization', plot_settings)
    e_net_list, spec_heat_cap_list, generation_list, ind_key_list, c_beta_list, c_beta_i_list \
        = convert_plot_data_lists(plot_data, thermalize=True, only_batch_num=0)
    simple_thermalization_plot(e_net_list, c_beta_list, plot_settings)


def main_annealing(plot_settings):
    '''
    Visualize chenge in network energy during annealing
    '''
    plot_data = load_files_anneal('energies_annealing', plot_settings)
    e_net_list, generation_list, ind_key_list \
        = convert_plot_data_lists(plot_data, thermalize=False, only_batch_num=0)
    simple_annealing_plot(e_net_list, plot_settings)


def simple_annealing_plot(e_net_list, plot_settings):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, e_nets in enumerate(e_net_list):
        if i > 2 and i < 6:
        # if i == 0:
            internal_steps = list(range(len(e_nets)))
            # plt.plot(internal_steps, e_nets, alpha = 1)
            plt.plot(internal_steps, e_nets , alpha=0.5, ms=0.3)

    plt.title(r'Annealing')
    plt.ylabel('$E_\mathrm{net}$')
    plt.xlabel('internal step')

    savefig_path = 'save/{}/figs/energies_annealing/'.format(plot_settings['sim_name'])
    mkdir(savefig_path)
    plt.savefig(savefig_path + 'energies_annealing_gen{}.png'.format(plot_settings['gen']))
    plt.show()


def simple_thermalization_plot(e_net_list, c_beta_list, plot_settings):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, e_nets in enumerate(e_net_list):
        if i == 0:
            internal_steps = list(range(len(e_nets)))
            plt.plot(internal_steps, e_nets, alpha = 1)

    plt.title(r'Network thermalization after initialization $c_\beta = {}$'.format(c_beta_list[0]))
    plt.ylabel('$E_\mathrm{net}$')
    plt.xlabel('internal step')

    savefig_path = 'save/{}/figs/energies_thermalization/'.format(plot_settings['sim_name'])
    mkdir(savefig_path)
    plt.savefig(savefig_path + 'energies_thermalization_gen{}_c_beta{}.png'
                .format(plot_settings['gen'], np.round(c_beta_list[0])))
    plt.show()


def load_files_thermalize(energies_folder_name, plot_settings):
    '''
    Load files for thermalization plot
    :param energies_folder_name: name of folder (not path), that energies files are saved in
    :return: e_net_list, spec_heat_cap_list, generation_list, ind_key_list, c_beta_list, c_beta_i_list
    '''
    sim_path = 'save/{}/'.format(plot_settings['sim_name'])
    load_requires_dict = {'c_beta_i': plot_settings['c_beta_i']}
    load_path = '{}{}/gen{}'.format(sim_path, energies_folder_name, str(plot_settings['gen']).zfill(6))
    file_list = load_files_ints_after_substr(load_path, 'energies',
                                 load_requires_dict, only_load_one_file=False)
    return file_list


def load_files_anneal(energies_folder_name, plot_settings):
    '''
    Load files for annealing plot
    '''
    sim_path = 'save/{}/'.format(plot_settings['sim_name'])
    load_path = '{}{}/gen{}/'.format(sim_path, energies_folder_name, str(plot_settings['gen']).zfill(6))
    file_list = load_all_files_that_include('energies', load_path)
    return file_list


def convert_plot_data_lists(plot_data, thermalize=True, only_batch_num=None):
    '''

    :param plot_data: e_net_list, spec_heat_cap_batch, generation, ind_key, c_beta, c_beta_i
    :param thermalize: Set true for thermalize plot and false for annealing plot
    :param only_batch_num: Only plot one particular batch
    :return:
    '''
    # plot_data:
    if only_batch_num is None:
        e_net_list = [[tensor.tolist() for tensor in data[0]] for data in plot_data]
        if thermalize:
            spec_heat_cap_list = [data[1].tolist() for data in plot_data]
    else:
        e_net_list = extract_e_net_list_one_batch(plot_data, only_batch_num)
        if thermalize:
            spec_heat_cap_list = [data[1][only_batch_num].tolist() for data in plot_data]
    generation_list = [data[2] for data in plot_data]
    ind_key_list = [data[3] for data in plot_data]
    if thermalize:
        c_beta_list = [data[4] for data in plot_data]
        c_beta_i_list = [data[5] for data in plot_data]
    if thermalize:
        return e_net_list, spec_heat_cap_list, generation_list, ind_key_list, c_beta_list, c_beta_i_list
    else:
        return e_net_list, generation_list, ind_key_list


def extract_e_net_list_one_batch(plot_data, only_batch_num):
    '''
    Get the network energy time series for one batch
    '''
    e_net_list = []
    for data in plot_data:
        e_net_internal_step = []
        for i in range(len(data[0])):
            e_net_internal_step.append(data[0][i].tolist()[only_batch_num])
        e_net_list.append(e_net_internal_step)
    return e_net_list


if __name__ == '__main__':
    '''
    PLOT SETTINGS
    This plots the series of network energies during the Monte-Carlo procedure to equlibrize the networks
    One plot that shows the energy series from which the heat capacity is calculated (by using the variance of the series)
    The other plot shows the energy series during the simulated annealing procedure that is used to find the energy minimum,
    at which the network is initialized for the heat capacity calculations.
    
    Practical set-up for playing around...
    The figures are saved in each according simulation folder
    '''

    plot_settings = {}
    # Choose simulation folder:
    plot_settings['sim_name'] = 'sim_20220125-122734_Example_Run_short'
    # PLot generation:
    plot_settings['gen'] = 0
    # Index of the beta factor for which the series shall be created (value on x-axis of heat-capacity plots)
    plot_settings['c_beta_i'] = 50
    # Make the figures:
    main_thermalization(plot_settings)
    main_annealing(plot_settings)




