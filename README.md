# The evolution of the dynamical regime in the context of a novelty-based fitness metric

## Project
Ising-neural networks were implemented in a state-of-the-art neuro-evolutionary algorithm to solve standardized machine 
learning tasks such as the pole balancing game and is compatible with Open-AI Gym environments.
The goal of the project is to examine the criticality hypothesis with respect to a fitness function, that rewards novel behavior instead of 
performance with respect to the task objective. The criticality hypothesis states, that evolutionary dynamics press living 
systems close to the critical point. Closeness to criticality is associated with many adaptable and explorative 
behavioural patterns, that may be beneficial for living systems. In a previous objective/performance-based approach we found, that
agents performed optimally in a sub-critical regime close to criticality. However the distance to criticality of the 
optimal regime decreased for an increased task difficulty. As ever-novel behaviours are observed in natural evolution
we would like to know how the evolution of the agent's dynamical regime would be influenced by a novelty-based fitness
function compared to a traditional objective-based fitness function. 

## Technical implementation

The code is kept highly modular such that an experiment can be quickly assembled in the main file (an example main_cartpole.py)
and new features can be added easily. For efficiency, all network operations are implemented in PyTorch (on GPU or CPU), where an agent is evaluated batch-wise
in multiple environments simoultaneously.

The following set-ups can be chosen in main.py (for details check next section)
* The type of network (Ising-based or Recurrent), for Ising the heat capacity can be calculated to infer the dynamical regime
* The Open-AI Gym environment (CartPole and AcroBot tested)
* The environment evaluator (Novelty-based or traditional objective-based)
* The reporter (states what experiment data is saved for later processing/plotting and what is printed to the console)
* The config file, which defines the hyperparameters for the evolutionary algorithm NEAT
* The parameters of the global_dict give the settings for the current experiment

The controllers are encoded as standardized genomes, that are evolved by a modified version of NEAT-Python. 
The Project is based upon the package PyTorch-NEAT. It also usues a heavily modified version of NEAT-Python.

For the Ising networks the heat capacity, magnetic susceptibility and magnetization can be calculated for a broad range of
modified temperatures to infer the dynamical regime. The sensory input neurons of the networks are set to values, 
they experienced during the evaluation and initialized at an energy minimum, which is detected by a simulated annealing heuristic.
The calculations are implemented in parallel within PyTorch for efficiency.

The plotting folder contains plotting functions for heat-capacity and susceptibility divergence plots as well as 
magnetization plotting. Further the network energies can be plotted to visualize the annealing as well as the equlibrization
Metropolis-Hastings procedure, which is used to obtain the sensor outputs during evaluation. Further the fitness developement
can be visualized with respect to the species in NEAT.

## Setting up an experiment in main.py:

### Neural Controller
Choose a Network
Ising-based:
```
from pytorch_neat.ising_net import IsingNet as AgentNet
```
or Recurrent:
```
from pytorch_neat.recurrent_net import RecurrentNet as AgentNet
```

The following converts a genome into a PyTorch network:
```
net = AgentNet.create(genome, config, bs)
outputs = net.activate(some_array)
```

### Gym Environment:
Set up an Open-AI Gym environment. CartPole and Acrobot have been tested.
```
def make_env():
    return gym.make("CartPole-v0")
```
Or:
```
def make_env():
    return gym.make("Acrobot-v1")
```

### Environment evaluator:
Choose either a novelty-based or traditional objective-based environment evaluator
Implementing an environemnt evaluator works like this:
```
from pytorch_neat.multi_env_eval_save_inputs import MultiEnvEvaluator
```

And evaluate the agents:
```
    evaluator = MultiEnvEvaluator(
                 ...
                )
                
    def eval_genomes(genomes, ...):
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = evaluator.eval_genome(genome, ...)
```


### Installation:
Install preferably with miniforge but any conda build should do.
First install latest PyTorch build for your system (this version was installed in january 2024)
Delete all torch-related stuff from new_build_environemnt.yml
Install the rest from new_build_environemnt.yml
```angular2html
mamba create -n phdenv
mamba env update -n phdenv -f phdenv.yml
```
subsequently install ray and python-neat 0.92 using pip
```angular2html
pip install ray # In my tested build I accidentally used pip instead of pip3 to install ray, but pip3 should do, too. Installed ray version was 2.9.0, but the latest version should do
pip3 install neat-python==0.92
```

