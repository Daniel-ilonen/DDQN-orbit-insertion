# DDQN-orbit-insertion
## Summary
Using Deep Reinforcement Learning with Double Q-learning to teach a spaceship to achieve orbit in a simple orbital mechanics simulation

## Enviroment
The enviroment consists of a rocket, a planet, and an atmosphere. The atmosphere induces drag, and its density decreases linerly until 'space' is reached. 
The enviroment requires PyGame, but its definitions and return parameters should be similar to the ones used by openAI/Gym for ease of testing on other enviroments.
![Picture of Enviroment](/plots/Screenshot.png?raw=true "Enviroment")
## DDQN
To encourage the DDQN agent to reach orbit we use some reward function. A reward function based on $\theta*\rho^{-1}$ seems to work well, where $\theta$ is the angle traveled around the planet, and $\rho$ is the atmospheric density of the current position. 

### Hyperparameters
The hyperparameters include  
```
    params['replays_per_session']=10
    params['epsilon_max'] = 1.0
    params['epsilon_min'] = 0.01
    params['gamma'] = 0.95
    params['learning_rate'] = 0.001
    params['first_layer_size'] = 124    # neurons in the first layer
    params['second_layer_size'] = 64   # neurons in the second layer
    params['episodes'] = 20000
    params['memory_size'] = 10000
    params['batch_size'] = 128
```
where "replays_per_session" indicates how many times agent Q1 uses a batch to learn from memory after every episode before the Q1 weights are copied to Q2. 
Epsilon is right now set to $\epsilon=e^{ln(\epsilon_{min})/(replays_per_session*episodes*0.75)}$

## Evaluation
![Plot](/plots/Average_reward.png?raw=true "Training")