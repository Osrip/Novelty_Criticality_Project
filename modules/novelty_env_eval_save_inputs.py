
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import ray
import os
import time
from helper_functions.loading_saving import save_pickle_compressed
import torch


class MultiEnvEvaluatorNovelty:
    '''
    This Environment evaluator class evaluates the networks and also saves the data, that is transferred from the
    environment to the sensory input neurons to a .pickle file in the current simulation's save folder. The evaluation
    is done in parallel for each batch.

    It implements a novelty-based fitness function

    Attributes
    ---
    make_net: Function, that creates network
    activate_net: Function, that creates the outputs of the network given the inputs
    batch_size: int, that states the number of batches
    max_env_steps: maximal amount of time steps in the environment evaluation procedure
    save_dir: str, save directory of current simulation
    envs: list, of environment objects, one environment onject for each batch. Envs can be passed directly or a make_env
    function can be passed when an instance is initialized

    Methods
    ---
    eval_genome(self, genome, config, global_dict, debug=False, record=False, display_env=False, save_sensor=True,
                    suppress_input_saving=False)
        Evaluates the given genome in multiple environments parallely
    '''
    def __init__(self, make_net, activate_net, global_dict, batch_size=1, max_env_steps=None, make_env=None, envs=None):
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            self.envs = envs
        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.save_dir = global_dict['save_dir']


    # Make similar function for Novelty search!!!!
    def eval_genome(self, genome, config, global_dict, debug=False, record=False, display_env=False, save_sensor=True,
                    suppress_input_saving=False):
        '''
        Evaluates the given genome in multiple environments parallely
        '''
        net = self.make_net(genome, config, global_dict, self.batch_size)

        fitnesses = np.zeros(self.batch_size)

        states = [env.reset()[0] for env in self.envs]
        dones = [False] * self.batch_size

        if record:
            video_env = self.envs[0]
            video_path = 'videos_{}/'.format(time.strftime("%Y%m%d-%H%M%S"))
            video_name = 'video_{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S"))
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            video_recorder = VideoRecorder(video_env, video_path+video_name, enabled=record)

        step_num = 0
        # Time loop
        all_states = []
        max_env_steps_gym = self.envs[0].spec.max_episode_steps
        if self.max_env_steps is not None:
            if max_env_steps_gym > self.max_env_steps:
                max_env_steps_gym = self.max_env_steps

        all_states_mat = np.zeros((self.batch_size, max_env_steps_gym, net.n_inputs))
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
                actions = actions.astype('int64')
            else:
                actions = self.activate_net(net, states)
                actions = actions.astype('int64')
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if display_env:
                    env.render()
                if record and i == 0:
                    # env.unwrapped.render()
                    env.render()
                    video_recorder.capture_frame()
                if not done:
                    # Weirdly there is an empty dict at the end as an output of the step function
                    state, reward, done, _ = env.step(action)[0:-1]

                    fitnesses[i] += reward
                    if not done:
                        for j, input in enumerate(state):
                            all_states_mat[i, step_num - 1, j] = input
                        states[i] = state
                    dones[i] = done
            # TODO: This used to be two intends further right, now correct?
            for state in states:
                all_states.append(state)


                    # States:
                    # 1 position of the cart on the track
                    # 2 angle of the pole with the vertical
                    # 3 cart velocity
                    # 4 rate of change of the angle

            if all(dones):
                break
        if record:
            video_recorder.close()
            video_recorder.enabled = False

        if save_sensor:
            save_name = 'sensor_ind{}.pickle'.format(genome.key)
            curr_save_dir = '{}sensor_inputs/gen{}/'.format(self.save_dir, str(genome.generation).zfill(6))
            if not os.path.exists(curr_save_dir):
                os.makedirs(curr_save_dir)
            save_pickle_compressed(curr_save_dir+save_name, torch.DoubleTensor(all_states))


        return sum(fitnesses) / len(fitnesses), all_states_mat


