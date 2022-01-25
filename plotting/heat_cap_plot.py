from helper_functions.loading_saving import load_pickle_compressed
from helper_functions.loading_saving import load_all_files_that_include_as_dict
from helper_functions.loading_saving import mkdir
import matplotlib.pyplot as plt
import numpy as np


def main(plot_settings):
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': plot_settings['font_size'], 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': plot_settings['font_size']})

    sim_name = plot_settings['sim_name']
    c_betas, all_heat_caps_c_beta = load_data(sim_name, plot_settings)
    plot(c_betas, all_heat_caps_c_beta, plot_settings)


def load_data(sim_name, plot_settings):
    if plot_settings['susceptibility']:
        save_heat_cap_path = 'save/{}/susceptibility_data/gen{}/'.format(sim_name, plot_settings['gen'])
        include_name = 'susceptibility_ind'
    elif plot_settings['magnetization']:
        save_heat_cap_path = 'save/{}/magnetization_data/gen{}/'.format(sim_name, plot_settings['gen'])
        include_name = 'magnetization_ind'
    else:
        save_heat_cap_path = 'save/{}/heat_cap_data/gen{}/'.format(sim_name, plot_settings['gen'])
        include_name = 'heat_cap_ind'

    # save_heat_name = 'heat_cap_ind_{}.pickle'.format(genome.key)
    all_heat_caps_c_beta = load_all_files_that_include_as_dict(include_name, save_heat_cap_path)
    save_beta_name = 'c_betas.pickle'
    c_betas = load_pickle_compressed('{}{}'.format(save_heat_cap_path, save_beta_name))
    return c_betas, all_heat_caps_c_beta


def plot(c_betas, all_heat_caps_c_beta, plot_settings):
    '''
    Create heat capacity plots
    '''
    for ind in plot_settings['plot_inds']:

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, (key, heat_caps) in enumerate(all_heat_caps_c_beta.items()):
            if i==ind or 'all' in str(ind):
                if 'all' in str(ind):
                    plt.plot(c_betas, np.array(heat_caps), alpha = 0.3)
                else:
                    plt.scatter(c_betas, np.array(heat_caps), s=0.5, marker='X')

        ax.set_xscale('log')
        # ax.set_yscale('log')
        if ind == 'all_lim':
            plt.ylim(-0.1, 2.5)
            plt.ylim(-0.03, 0.35)
        # plt.xlim(10**-1, 10**3)
        plt.xlabel(r'$c_\beta$')

        if plot_settings['susceptibility']:
            plt.ylabel(r'Susceptibility')
            savefig_path = 'save/{}/figs/susceptibility/'.format(plot_settings['sim_name'])
            savefig_name = 'susceptibility_gen{}_ind_{}.png'.format(plot_settings['gen'], ind)
        elif plot_settings['magnetization']:
            plt.ylabel(r'Magnetization')
            savefig_path = 'save/{}/figs/magnetization/'.format(plot_settings['sim_name'])
            savefig_name = 'magnetization_gen{}_ind_{}.png'.format(plot_settings['gen'], ind)
        else:
            plt.ylabel(r'$C_H/N$')
            savefig_path = 'save/{}/figs/heat_capacity/'.format(plot_settings['sim_name'])
            savefig_name = 'heat_capacity_gen{}_ind_{}.png'.format(plot_settings['gen'], ind)
        mkdir(savefig_path)
        plt.savefig(savefig_path+savefig_name, bbox_inches='tight', dpi=150)
        plt.show()


if __name__=='__main__':
    '''
    PLOT SETTINGS
    This creates heat capacity plots for the Ising-based agent controllers
    (OR alternatively magnetic susceptibility plots OR magnetization)
    Practical set-up for playing around...
    The figures are saved in each according simulation folder
    '''
    # List all folder names of simulation runs, that you wish to create plots for:
    sim_names = ['sim_20220125-122734_Example_Run_short']
    for sim_name in sim_names:
        plot_settings = {}
        plot_settings['sim_name'] = sim_name
        # List all individuals, that shall be plotted (each entry creates a figure per simulation):
        # ('all' plots all in one, 'all_lim' plots all in on with limeted y-bouns)
        plot_settings['plot_inds'] = ['all', 'all_lim', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Create plot for generation:
        plot_settings['gen'] = 0
        # Font size of the labels
        plot_settings['font_size'] = 15
        # Plot susceptibility instead of heat capacity:
        plot_settings['susceptibility'] = False
        # PLot magnetization instead of heat capacity:
        plot_settings['magnetization'] = False
        main(plot_settings)
