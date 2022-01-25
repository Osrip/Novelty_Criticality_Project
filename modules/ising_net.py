
import torch
import numpy as np
from .activations import sigmoid_activation
from helper_functions.loading_saving import load_pickle_compressed
from helper_functions.loading_saving import save_pickle_compressed
from helper_functions.loading_saving import mkdir
import itertools

torch.nn.parallel.DistributedDataParallel


class IsingNet():
    '''
    Network class that implements a statistical neural network based upon the rules of the Ising model.
    Each neuron can be in one of two states [-1,1] with the exception of the sensory input neurons, which
    have float values and remain fixed.

    Attributes
    ---
    n_inputs: int, number of sensory input neurons
    n_hidden: int, number of hidden neurons
    n_outputs: int, number of output neurons

    beta: float, inverse temeperatur of the network (defined in genome and suject to evolution. The NEAT-Python package
    had to be modified in order to be able to process this additional parameter)
    genome: genome object, that defines the connectivity as well as beta of the network
    global_dict: dict, includes settings of the current experiment
    dev: str, device, that pytorch calculations are done upon ('cuda': GPU, 'cpu': CPU)

    n_internal_steps: int, number of iterations in outer loop of metropolis-hastings algorithm (number of times each neuron
    is checked for a potential spin-flip)
    dtype: dtype, Data type used for PyTorch tensors

    activations are not used here. They are still here to keep the notations compatible with the RecurrentNet class

    Methods
    ---
    reset(self, batch_size=1)
        Initialize the network states randomly (except sensory input neurons)

    activate(self, inputs, record_sensors=True)
        Takes sensor inputs, then calculates the network states at equilibrium using a Metropolis-Hastings Heuristic.
        The states of the output neurons are then returned. The sensory inputs remain fixed during the equilibrization.

    calculate_heat_capacity(self, n_internal_heat_cap, exp_c_beta_max, exp_c_beta_min, num_c_beta,
                                magnetic_susceptibility=False, magnetization=False)
        Calculates the heat capacity for the current network (OR alternatively magnetic suscepibility OR magnetization)
        for multiple values of the beta factor c_beta (given by exp_c_beta_max, exp_c_beta_min, num_c_beta). This data can be used to
        estimate the critical temperature. Normalization by number of neurons (specific heat capacity)

    calculate_heat_capacity_one_beta(self, c_beta, c_beta_i, n_internal_heat_cap, states_init, J, size, inputs,
                                         magnetic_susceptibility=False, magnetization=False):
        Calculates the heat capacity (alternatively magnetic susceptibility OR magnetization) for one beta value
        (helper for calculate_heat_capacity() )
        More info in Docstring of calculate_heat_capacity()

    create_J(self)
        Creates network connectivity matrix J from class attributes

    randomly_initialize_states_fixed_sensors(self, inputs)
        Creates initial random states with fixed sensor values (inputs)

    brute_force_min_energy(self, inputs, states, J)
        Brute force calculation of minimal network energy (extremely computationally expensive)

    simulated_annealing(self, states, J, n_internal_anneal, exp_beta_min_anneal, exp_beta_max_anneal,
                            return_overall_minimum=False)
        Heuristic to calculate network's energy minimum. Runs the Metropolis-Hastings algorithm while the network's
        inverse temperature beta is increased. Takes batch dimensions.

    create(genome, config, global_dict, batch_size=1, activation=sigmoid_activation,
               prune_empty=False, use_current_activs=False, n_internal_steps=1)
        Creates an Ising network object (static method)
    '''
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 input_to_hidden, hidden_to_hidden, output_to_hidden,
                 input_to_output, hidden_to_output, output_to_output,
                 hidden_responses, output_responses,
                 hidden_biases, output_biases, genome, global_dict,
                 batch_size=1,
                 use_current_activs=False,
                 activation=sigmoid_activation,
                 n_internal_steps=10,
                 dtype=torch.float64, beta=1):
        torch.set_num_threads(global_dict['cores'])

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.beta = beta
        self.genome = genome
        self.global_dict = global_dict
        self.genome.input_record = []

        self.use_current_activs = use_current_activs
        self.activation = activation
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype

        self.dev = global_dict['device']

        if n_hidden > 0:
            self.input_to_hidden = dense_from_coo(
                (n_hidden, n_inputs), input_to_hidden, self.dev, dtype=dtype)
            self.hidden_to_hidden = dense_from_coo(
                (n_hidden, n_hidden), hidden_to_hidden, self.dev, dtype=dtype)
            self.output_to_hidden = dense_from_coo(
                (n_hidden, n_outputs), output_to_hidden, self.dev, dtype=dtype)
            self.hidden_to_output = dense_from_coo(
                (n_outputs, n_hidden), hidden_to_output, self.dev, dtype=dtype)
        self.input_to_output = dense_from_coo(
            (n_outputs, n_inputs), input_to_output, self.dev, dtype=dtype)
        self.output_to_output = dense_from_coo(
            (n_outputs, n_outputs), output_to_output, self.dev, dtype=dtype)

        if n_hidden > 0:
            self.hidden_responses = torch.tensor(hidden_responses, dtype=dtype, device=self.dev)
            self.hidden_biases = torch.tensor(hidden_biases, dtype=dtype, device=self.dev)

        self.output_responses = torch.tensor(
            output_responses, dtype=dtype, device=self.dev)
        self.output_biases = torch.tensor(output_biases, dtype=dtype, device=self.dev)

        self.reset(batch_size)

    def reset(self, batch_size=1):
        '''
        Initialize the network states randomly (except sensory input neurons)
        '''

        if self.n_hidden > 0:
            self.activs = (torch.rand(batch_size, self.n_hidden, device=self.dev) > 0.5).type(self.dtype) * 2.0 - 1.0
            # self.activs = torch.zeros(
            #     batch_size, self.n_hidden, dtype=self.dtype)
        else:
            self.activs = None

        self.outputs = (torch.rand(batch_size, self.n_outputs, device=self.dev) > 0.5).type(self.dtype) * 2.0 - 1.0
        # self.outputs = torch.zeros(
        #     batch_size, self.n_outputs, dtype=self.dtcreateype)

    def activate(self, inputs, record_sensors=True):
        '''
        Takes sensor inputs, then calculates the network states at equilibrium using a Metropolis-Hastings Heuristic.
        The states of the output neurons are then returned. The sensory inputs remain fixed during the equilibrization.

        The function is able to handle an arbitrary amount of batch-dimensions for performance optimization,
         with each batch dimension representing a parallel evaluation of the current network in different environments

        :param inputs: States of network's input neurons
        :param record_sensors [boolean]: if True, all sensor values are saved. Saved values are used as initialization
         for heat capacity calculations such that the heat capacity divergence according to the network's state
         during evaluation in the gym environment
        :return: States of output neurons
        '''
        # Record sensor inputs
        if record_sensors:
            if len(np.shape(inputs)) < 2:
                self.genome.input_record.append(inputs)
            else:
                for in_batch in inputs:
                    self.genome.input_record.append(in_batch)
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=self.dtype, device=self.dev)

            # Create states according to attributes:
            if self.n_hidden > 0:
                states = torch.cat([inputs, self.activs, self.outputs], dim=1)
            else:
                states = torch.cat([inputs, self.outputs], dim=1)

            # if there is no batch dimension, add an empty dim.
            if len(inputs.shape) < 2:
                states = states.unsqueeze(0)

            # Create Connectivity J according to attributes
            size = self.n_inputs + self.n_hidden + self.n_outputs
            J = torch.zeros(size, size, dtype=self.dtype, device=self.dev)
            input_slice = slice(0, self.n_inputs)
            hidden_slice = slice(self.n_inputs, -self.n_outputs)
            output_slice = slice(-self.n_outputs, None)

            if self.n_hidden > 0:
                J[hidden_slice, input_slice] = self.input_to_hidden
                J[hidden_slice, output_slice] = self.output_to_hidden
                J[hidden_slice, hidden_slice] = self.hidden_to_hidden

                J[output_slice, hidden_slice] = self.hidden_to_output

            J[output_slice, input_slice] = self.input_to_output
            J[output_slice, output_slice] = self.output_to_output

            # In case of symmetric edges, make connectivity J symmetric
            if self.global_dict['make_edges_symmetric']:
                J = (J + J.transpose(1, 0)) / 2

            # METROPOLIS-HASTINGS ALGORITHM to calculate the states of the Ising network at equilibrium.
            # Batch-wise parallelization of multiple environment evaluations of the same network to optimize
            # performance. Each batch dimension corresponds to the state matrix of the network in a different
            # environment.
            n_updatable_neurons = size - self.n_inputs
            random_vars = np.random.rand(self.n_internal_steps, n_updatable_neurons)
            # OUTER LOOP: One iteration (internal Ising step) of loop checks for spin flip in each neuron of the network
            for i in range(self.n_internal_steps):
                # INNER LOOP: Iterate through every neuron / site of the Ising network in a randomly perturbed order
                # to check for spin-flip
                perms = np.random.permutation(np.arange(self.n_inputs, size))
                for j, perm in enumerate(perms):
                    # e_diff is the network's energy difference in case of a spin flip
                    # (calculated parallely for each batch)
                    e_diff = 2 * states[:, perm] * torch.matmul(states, J[perm, :])
                    rand = random_vars[i, j]
                    #  bool_tensor states batch-wise, whether spin-flip is executed
                    # (spin-flip probability is calculated and spin-flip is executed according to random variable)
                    bool_tensor = self.beta * e_diff < np.log(1.0 / rand -1)
                    # Flip states at current neuron according to bool_tensor for each batch
                    flip_tensor = torch.zeros(bool_tensor.shape, device=self.dev)
                    flip_tensor[bool_tensor==0] = 1.0
                    flip_tensor[bool_tensor==1] = -1.0
                    states[:, perm] = states[:, perm] * flip_tensor.double()

            self.activs = states[:, hidden_slice]
            self.outputs = states[:, output_slice]
        return self.outputs

    def calculate_heat_capacity(self, n_internal_heat_cap, exp_c_beta_max, exp_c_beta_min, num_c_beta,
                                magnetic_susceptibility=False, magnetization=False):
        '''
        Calculates the heat capacity for the current network (OR alternatively magnetic suscepibility OR magnetization)
        for multiple values of the beta factor c_beta (given by exp_c_beta_max, exp_c_beta_min, num_c_beta). This data can be used to
        estimate the critical temperature. Normalization by number of neurons (specific heat capacity)

        Details of the algorithm: Each batch represents a repetition of the heat capacity calculation of which
        the mean is taken. Each batch is initialized with sensory input values, that have been recorded previously
        during the evaluation in the gym environment. This way the heat capacity values are representative of the energy
        minima, that the network equilibrizes towards during the evaluation. Subsequently the energy minimum of the
        network is found by a simulated annealing heuristic (or alternatively a brute-force approach). For the
        heat-capacity calculation the network is initialized at the energy minimum for the current recorded sensory input,
        which remains fixed. During each iteration of the outer loop in the Metropolis-Hastings procedure, the network
        energy is calculated. The heat capacity is calculated from the variance of the network energy.

        :param n_internal_heat_cap: Number of internal steps (iterations outer loop in Metropolis Algorithm) for which
        the network energy is calculated. From the variance of the energies calculated this way, the heat capacity is
        calculated
        :param exp_c_beta_max: exp upper bound for c_beta
        :param exp_c_beta_min: exp lower bound for c_beta
        :param num_c_beta: number of c_betas to calculate the heat capacity for
        :param magnetic_susceptibility: if True, calculate magnetic susceptibility instead of heat capacity
        :param magnetization: if True, calculate magnetization instead of heat capacity
        :return: c_betas, heat_caps_c_beta
        '''

        c_beta_exps = np.linspace(exp_c_beta_min, exp_c_beta_max, num_c_beta)
        c_betas = [10**c_beta_exp for c_beta_exp in c_beta_exps]
        heat_caps_c_beta = []

        save_name = 'sensor_ind{}.pickle'.format(self.genome.key)
        curr_save_dir = '{}sensor_inputs/gen{}/'.format(self.global_dict['save_dir'],
                                                        str(self.genome.generation).zfill(6))
        inputs = load_pickle_compressed(curr_save_dir + save_name)
        inputs_sampled = [inputs[rand] for rand in np.random.randint(len(inputs),
                                                                     size=self.global_dict['num_batches_heat_cap'])]
        inputs = torch.stack(inputs_sampled)
        J, size = self.create_J()
        if self.global_dict['make_edges_symmetric']:
            J = (J + J.transpose(1, 0)) / 2

        # Only calculate low energy states once for all c_betas, saves time but could lead to bias in case heuristic
        # does not find good minimum
        states_init = self.randomly_initialize_states_fixed_sensors(inputs)

        if self.global_dict['brute_force_init']:

            states_init = self.brute_force_min_energy(inputs, states_init, J)
        else:

            states_init = self.simulated_annealing(states_init, J, self.global_dict['n_internal_anneal'],
                                              self.global_dict['exp_beta_min_anneal'],
                                              self.global_dict['exp_beta_max_anneal'],
                                                   return_overall_minimum=False)

        for c_bet_i, c_beta in enumerate(c_betas):
            spec_heat_caps = self.calculate_heat_capacity_one_beta(c_beta, c_bet_i, n_internal_heat_cap, states_init, J, size,
                                                                   inputs, magnetic_susceptibility, magnetization)
            # Taking mean over all batches
            spec_heat_cap = float(torch.mean(spec_heat_caps))

            heat_caps_c_beta.append(spec_heat_cap)
        return c_betas, heat_caps_c_beta

    def calculate_heat_capacity_one_beta(self, c_beta, c_beta_i, n_internal_heat_cap, states_init, J, size, inputs,
                                         magnetic_susceptibility=False, magnetization=False):
        '''
        Calculates the heat capacity (alternatively magnetic susceptibility OR magnetization) for one beta value
        (helper for calculate_heat_capacity() )
        More info in Docstring of calculate_heat_capacity()
        '''

        beta_new = self.beta * c_beta
        states = states_init

        e_mean = torch.zeros(len(inputs), dtype=self.dtype, device=self.dev)
        e2_mean = torch.zeros(len(inputs), dtype=self.dtype, device=self.dev)

        if self.global_dict['save_thermalization']:
            e_net_list = []

        with torch.no_grad():
            # calculating heat capacity for all recorded states of sensory input neurons parallely in batches

            n_updatable_neurons = size - self.n_inputs

            # Metropolis-Hastings Algorithm. Look into activate() for well commented version.
            # Had to be re-written heat-capacity specific in a modified way for for optimal performance
            random_vars = np.random.rand(n_internal_heat_cap, n_updatable_neurons)
            ct = 0
            for i in range(n_internal_heat_cap):
                perms = np.random.permutation(np.arange(self.n_inputs, size))
                for j, perm in enumerate(perms):
                    e_diff = 2 * states[:, perm] * torch.matmul(states, J[perm, :])
                    rand = random_vars[i, j]
                    bool_tensor = beta_new * e_diff < np.log(1.0 / rand -1)
                    flip_tensor = torch.zeros(bool_tensor.shape, device=self.dev)
                    flip_tensor[bool_tensor==0] = 1.0
                    flip_tensor[bool_tensor==1] = -1.0
                    states[:, perm] = states[:, perm] * flip_tensor.double()
                    # Save network energy at every internal step of outer metropolis loop. Will decrease performance
                    # and is not required for varaince calculation (as only mean and mean^2 has to be known)
                    if self.global_dict['save_thermalization']:
                        # This takes a batch-wise dot product between J and the states.
                        # Checked this on paper (Note to author: calculations saved in icloud notes, search pytorch)
                        e_net = -torch.sum((torch.matmul(states, J) * states), dim=1)
                        e_net_list.append(e_net)

                if i > self.global_dict['internal_steps_before_recording_heat_cap']:
                    ct += 1

                    if magnetic_susceptibility or magnetization:
                        if self.global_dict['include_sensors_in_susceptibility']:
                            e_net = torch.mean(states, 1)
                        else:
                            e_net = torch.mean(states[:, self.n_inputs:], 1)
                    else:
                        e_net = -torch.sum((torch.matmul(states, J) * states), dim=1)
                    e_mean += e_net
                    e2_mean += e_net * e_net

            if not magnetization:
                # For sepc heat cap not divided by total numer of neurons but just by number of thermalizable neurons
                # (thus input not included)
                e_mean = e_mean / ct
                e2_mean = e2_mean / ct
                spec_heat_caps = beta_new ** 2 * (e2_mean - e_mean * e_mean) / (self.n_hidden + self.n_outputs)
            else:
                # Just take mean of magnetizations for magnetization
                spec_heat_caps = e_mean / ct

            if self.global_dict['save_thermalization']:
                save_energies('energies_thermalization', e_net_list, spec_heat_caps, self.genome.generation,
                              self.genome.key, c_beta, c_beta_i, self.global_dict)
            return spec_heat_caps

    def create_J(self):
        '''
        Creates network connectivity matrix J from class attributes
        :return: J, size
        '''
        size = self.n_inputs + self.n_hidden + self.n_outputs
        J = torch.zeros(size, size, dtype=self.dtype, device=self.dev)
        input_slice = slice(0, self.n_inputs)
        hidden_slice = slice(self.n_inputs, -self.n_outputs)
        output_slice = slice(-self.n_outputs, None)

        if self.n_hidden > 0:
            J[hidden_slice, input_slice] = self.input_to_hidden
            J[hidden_slice, output_slice] = self.output_to_hidden
            J[hidden_slice, hidden_slice] = self.hidden_to_hidden
            J[output_slice, hidden_slice] = self.hidden_to_output

        J[output_slice, input_slice] = self.input_to_output
        J[output_slice, output_slice] = self.output_to_output

        return J, size

    def randomly_initialize_states_fixed_sensors(self, inputs):
        '''
        Creates initial random states with fixed sensor values (inputs)
        :return: states
        '''
        self.reset(batch_size=inputs.shape[0])
        if self.n_hidden > 0:
            states = torch.cat([inputs, self.activs, self.outputs], dim=1)
        else:
            states = torch.cat([inputs, self.outputs], dim=1)
        # if there is no batch dimension, add an empty dim.
        if len(inputs.shape) < 2:
            states = states.unsqueeze(0)
        return states

    def brute_force_min_energy(self, inputs, states, J):
        '''
        Brute force calculation of minimal network energy (extremely computationally expensive)
        :return: states
        '''

        num_hidden_output = states.shape[1] - inputs.shape[1]
        permutated_states = list(itertools.product([-1, 1], repeat=num_hidden_output))
        permutated_states_w_inputs = torch.tensor(
            [[input_batch + list(permut) for permut in permutated_states] for input_batch in inputs.tolist()]
            , dtype=self.dtype, device=self.dev)
        e_nets = -torch.sum((torch.matmul(permutated_states_w_inputs, J) * permutated_states_w_inputs), dim=2)
        e_nets_argmin = torch.argmin(e_nets, dim=1)

        min_energy_states = permutated_states_w_inputs[np.arange(len(e_nets_argmin)), e_nets_argmin]  # permutated_states_w_inputs[np.arange(len(torch.argmin(e_nets, dim=1))),torch.argmin(e_nets, dim=1)]

        return min_energy_states

    def simulated_annealing(self, states, J, n_internal_anneal, exp_beta_min_anneal, exp_beta_max_anneal,
                            return_overall_minimum=False):
        '''
        Heuristic to calculate network's energy minimum. Runs the Metropolis-Hastings algorithm while the network's
        inverse temperature beta is increased. Takes batch dimensions.

        :param n_internal_anneal: number of internal steps (outer loop iterations in Metropolis algorithm)
        :param exp_beta_min_anneal: exp inverse temperature that the annealing procedure is started at
        :param exp_beta_max_anneal: exp inverse temperature that annealing procedure is ended at
        :param return_overall_minimum: if True: saves all states and corresponding energies during annealing.
        The overall energy minimum is returned, If false, return state matrix at the end of the annealing process
        :return: state matrix at energy minimum
        '''
        anneal_beta_exps = np.linspace(exp_beta_min_anneal, exp_beta_max_anneal, n_internal_anneal)
        anneal_betas = [10**beta_exp for beta_exp in anneal_beta_exps]
        # TODO: Use anneals betas directly as beta (currrently implemented)
        #  or use as beta factor and multiply with net's beta value??
        size = self.n_inputs + self.n_hidden + self.n_outputs
        n_updatable_neurons = size - self.n_inputs
        random_vars = np.random.rand(n_internal_anneal, n_updatable_neurons)
        if self.global_dict['save_thermalization'] or return_overall_minimum:
            e_net_list = []
        if return_overall_minimum:
            states_list = []

        # Metropolis-Hastings Algorithm. Look into activate() for well commented version.
        # Had to be re-written simulated-annealing specific for optimal performance
        for i, anneal_beta in enumerate(anneal_betas):
            perms = np.random.permutation(np.arange(self.n_inputs, size))
            for j, perm in enumerate(perms):
                e_diff = 2 * states[:, perm] * torch.matmul(states, J[perm, :])
                rand = random_vars[i, j]
                bool_tensor = anneal_beta * e_diff < np.log(1.0 / rand -1)
                flip_tensor = torch.zeros(bool_tensor.shape)
                flip_tensor[bool_tensor==0] = 1.0
                flip_tensor[bool_tensor==1] = -1.0
                states[:, perm] = states[:, perm] * flip_tensor.double()

                if self.global_dict['save_thermalization'] or return_overall_minimum:
                    e_net = -torch.sum((torch.matmul(states, J) * states), dim=1)
                    e_net_list.append(e_net)
                if return_overall_minimum:
                    states_list.append(states)
        if self.global_dict['save_thermalization']:
            save_energies('energies_annealing', e_net_list, None, self.genome.generation,
                          self.genome.key, None, None, self.global_dict, anneal=True)

        if return_overall_minimum:
            states_list[np.argmin(e_net_list)]
        else:
            return states

    @staticmethod
    def create(genome, config, global_dict, batch_size=1, activation=sigmoid_activation,
               prune_empty=False, use_current_activs=False, n_internal_steps=1):
        '''
        Creates an Ising network object
        :return: IsingNet object
        '''

        from neat.graphs import required_for_output

        genome_config = config.genome_config
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys, genome.connections)
        if prune_empty:
            nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
                set(genome_config.input_keys))

        input_keys = list(genome_config.input_keys)
        hidden_keys = [k for k in genome.nodes.keys()
                       if k not in genome_config.output_keys]
        output_keys = list(genome_config.output_keys)

        hidden_responses = [genome.nodes[k].response for k in hidden_keys]
        output_responses = [genome.nodes[k].response for k in output_keys]

        hidden_biases = [genome.nodes[k].bias for k in hidden_keys]
        output_biases = [genome.nodes[k].bias for k in output_keys]

        if prune_empty:
            for i, key in enumerate(output_keys):
                if key not in nonempty:
                    output_biases[i] = 0.0

        n_inputs = len(input_keys)
        n_hidden = len(hidden_keys)
        n_outputs = len(output_keys)

        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}

        def key_to_idx(key):
            if key in input_keys:
                return input_key_to_idx[key]
            elif key in hidden_keys:
                return hidden_key_to_idx[key]
            elif key in output_keys:
                return output_key_to_idx[key]

        input_to_hidden = ([], [])
        hidden_to_hidden = ([], [])
        output_to_hidden = ([], [])
        input_to_output = ([], [])
        hidden_to_output = ([], [])
        output_to_output = ([], [])

        for conn in genome.connections.values():
            if not conn.enabled:
                continue

            i_key, o_key = conn.key
            if o_key not in required and i_key not in required:
                continue
            if prune_empty and i_key not in nonempty:
                print('Pruned {}'.format(conn.key))
                continue

            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)

            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = input_to_hidden
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hidden_to_hidden
            elif i_key in output_keys and o_key in hidden_keys:
                idxs, vals = output_to_hidden
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = input_to_output
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hidden_to_output
            elif i_key in output_keys and o_key in output_keys:
                idxs, vals = output_to_output
            else:
                raise ValueError(
                    'Invalid connection from key {} to key {}'.format(i_key, o_key))

            idxs.append((o_idx, i_idx))  # to, from
            vals.append(conn.weight)

        return IsingNet(n_inputs, n_hidden, n_outputs,
                        input_to_hidden, hidden_to_hidden, output_to_hidden,
                        input_to_output, hidden_to_output, output_to_output,
                        hidden_responses, output_responses,
                        hidden_biases, output_biases, genome, global_dict,
                        batch_size=batch_size,
                        activation=activation,
                        use_current_activs=use_current_activs,
                        n_internal_steps=n_internal_steps)


def save_energies(folder_name, e_net_list, spec_heat_cap_batch, generation, ind_key, c_beta, c_beta_i, global_dict,
                  anneal=False):
    '''
    Saves the network energy series during monte-carlo procedure for potential plotting
    '''
    save_dir = '{}/{}/gen{}/'.format(global_dict['save_dir'], folder_name, str(generation).zfill(6))
    if anneal:
        energies_save_name = 'energies_ind{}.pickle'.format(ind_key)
    else:
        energies_save_name = 'energies_ind{}_c_beta_i{}.pickle'.format(ind_key, c_beta_i)
    mkdir(save_dir)
    save_data = (e_net_list, spec_heat_cap_batch, generation, ind_key, c_beta, c_beta_i)
    save_pickle_compressed(save_dir+energies_save_name, save_data)


def dense_from_coo(shape, conns, device, dtype=torch.float64):
    mat = torch.zeros(shape, dtype=dtype, device=device)
    idxs, weights = conns
    if len(idxs) == 0:
        return mat
    rows, cols = np.array(idxs).transpose()
    mat[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = torch.tensor(
        weights, dtype=dtype, device=device)
    return mat

