import numpy as np
import itertools
import scipy.stats

def novelty_population(all_states_mats, genomes, global_dict):
    '''
    Calculate the mean behavioural distance in a population
    '''
    genome_behav_dists = np.zeros(len(genomes))
    for (ind_i, ind_j) in itertools.combinations(list(range(len(all_states_mats))),2):

        behav_dist = _novelty_inds(all_states_mats[ind_i], all_states_mats[ind_j], global_dict)

        genome_behav_dists[ind_i] += behav_dist
        genome_behav_dists[ind_j] += behav_dist
    avg_genome_behav_dists = genome_behav_dists / (len(genomes) - 1)
    return avg_genome_behav_dists


def _novelty_inds(ind1, ind2, global_dict):
    '''
    Calculate the mean behavioural distance between all batches of two individuals
    '''
    behav_dists = []
    for batch_i, batch_j in itertools.combinations(list(range(global_dict['batch_size'])), 2):
        behav_dist = _novelty_batches(ind1[batch_i,: ,:], ind2[batch_j,: ,:], global_dict)
        behav_dists.append(behav_dist)
    return np.average(behav_dists)

def _novelty_batches(batch1, batch2, global_dict):
    '''
    Calculate the behavioural distance between two batches (specific runs in a gym environment) based upon
    KL-divergence between sensor input data
    '''
    behav_dists = []
    for input_i in range(np.shape(batch1)[1]):
        behav_dist = _novelty_series(batch1[:, input_i], batch2[:, input_i], global_dict)
        behav_dists.append(behav_dist)
    return np.average(behav_dists)


def _novelty_series(series1, series2, global_dict):
    '''
    Calculate the behavioural distance between two time series of sensor input data by taking the KL-divergence
    between the input data distribution
    '''

    hist1, _ = np.histogram(series1, bins=global_dict['bin_size_compare_behav'], density=True)
    hist2, _ = np.histogram(series2, bins=global_dict['bin_size_compare_behav'], density=True)
    # kl divergence returns inf, when there is a zero in hist (division by zero)
    # Therefore replace all 0s with evry small values:
    # TODO: Is fifo.tiny too small and messing up the results
    #  np.finfo(np.dtype(hist1[0])).tiny
    hist1 = [0.0001 if val == 0 else val for val in hist1]
    hist2 = [0.0001 if val == 0 else val for val in hist2]
    # This should only return one value!!
    # KL-divergence --> wir nehmen kreuzentropie, da wir hier nicht die Wahrscheinlichkeitsfunktion P kennen müssen
    behav_dist = scipy.stats.entropy(hist1, hist2)
    return behav_dist
