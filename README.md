# Deep Q-Learning - Lunar Lander 🚀

---

````markdown

This project demonstrates how I built and trained a Deep Q-Network (DQN) agent to safely land a lunar lander on the moon using reinforcement learning. The environment is based on [OpenAI Gym’s LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/).

The goal is to teach an agent, through trial and error, to control the lander’s engines and navigate safely to the landing pad while maximizing cumulative rewards. Along the way, I implemented key RL techniques such as **experience replay** and **target networks** to stabilize training.

---

## Outline
- [1 - Import Packages](#1---import-packages)
- [2 - Hyperparameters](#2---hyperparameters)
- [3 - The Lunar Lander Environment](#3---the-lunar-lander-environment)
  - [3.1 Action Space](#31-action-space)
  - [3.2 Observation Space](#32-observation-space)
  - [3.3 Rewards](#33-rewards)
  - [3.4 Episode Termination](#34-episode-termination)
- [4 - Load the Environment](#4---load-the-environment)
- [5 - Interacting with the Environment](#5---interacting-with-the-environment)
- [6 - Deep Q-Learning](#6---deep-q-learning)
  - [6.1 Target Network](#61-target-network)
  - [6.2 Experience Replay](#62-experience-replay)
- [7 - Deep Q-Learning Algorithm](#7---deep-q-learning-algorithm)
- [8 - Updating the Network Weights](#8---updating-the-network-weights)
- [9 - Training the Agent](#9---training-the-agent)
- [10 - Watching the Agent in Action](#10---watching-the-agent-in-action)
- [11 - References](#11---references)

---

<a name="1"></a>
## 1 - Import Packages

I used the following libraries:
- **numpy** for scientific computing  
- **deque** for the replay buffer  
- **namedtuple** to store experience tuples  
- **gym** for the LunarLander-v2 environment  
- **tensorflow / keras** for building and training neural networks  
- **pyvirtualdisplay** & **PIL** for rendering the environment  
- **custom utils** (helper functions I wrote or reused for plotting, video creation, epsilon decay, etc.)

---

<a name="2"></a>
## 2 - Hyperparameters

Some key hyperparameters for training:
```python
MEMORY_SIZE = 100_000     # replay buffer size
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 4  # frequency of weight updates
````

---

<a name="3"></a>

## 3 - The Lunar Lander Environment

The **Lunar Lander** task challenges the agent to land between two flags at coordinate `(0,0)`. The lander starts at the top with random velocity and unlimited fuel. The environment is considered *solved* when the agent averages **200 points** over 100 episodes.

<p align="center">
  <img src="images/lunar_lander.gif" width="50%">
</p>

### 3.1 Action Space

The agent has 4 discrete actions:

* `0`: Do nothing
* `1`: Fire right engine
* `2`: Fire main engine
* `3`: Fire left engine

### 3.2 Observation Space

The state vector has 8 values:

* Position `(x, y)`
* Velocities `(x_dot, y_dot)`
* Angle `θ` and angular velocity
* Two booleans for leg-ground contact

### 3.3 Rewards

Rewards encourage safe landings and discourage wasteful or dangerous actions:

* Closer to landing pad → higher reward
* Slower speed → higher reward
* Upright angle → higher reward
* +10 per leg on ground
* -0.03 per side engine fire
* -0.3 per main engine fire
* +100 for safe landing / -100 for crash

### 3.4 Episode Termination

Episodes end if:

* The lander crashes
* The lander moves outside `|x| > 1`
* The maximum step limit is reached

---

<a name="4"></a>

## 4 - Load the Environment

I used `gym.make("LunarLander-v2")` to load the environment and inspected its action and observation space.

---

<a name="5"></a>

## 5 - Interacting with the Environment

The agent interacts with the environment using the `.step(action)` loop:

1. Select action
2. Apply action → environment returns `(next_state, reward, done, info)`
3. Continue until `done`

---

<a name="6"></a>

## 6 - Deep Q-Learning

Since the state space is continuous, I approximated the Q-function with a neural network:

**Q-Network Architecture:**

* Input layer → state vector
* Hidden layer (64 units, ReLU)
* Hidden layer (64 units, ReLU)
* Output layer → Q-values for each action

### 6.1 Target Network

To avoid instability, I used a **target network** (`Q̂`) updated with soft updates:

```
w⁻ ← τw + (1 - τ)w⁻
```

### 6.2 Experience Replay

I stored transitions `(s, a, r, s’, done)` in a replay buffer and trained the network using random mini-batches, which breaks correlations and improves efficiency.

---

<a name="7"></a>

## 7 - Deep Q-Learning Algorithm

The algorithm steps:

1. Initialize Q-network and target network
2. For each episode:

   * Reset environment
   * Loop through timesteps:

     * Choose action (ε-greedy)
     * Take step, store experience
     * Sample batch from replay buffer
     * Compute targets and loss
     * Update Q-network
     * Soft-update target network
   * Decay ε

---

<a name="8"></a>

## 8 - Updating the Network Weights

I used **TensorFlow’s GradientTape** to compute gradients and update Q-network weights, with periodic soft updates to the target network.

---

<a name="9"></a>

## 9 - Training the Agent

I trained for up to **2000 episodes**, with each capped at 1000 timesteps. The agent solved the environment when its rolling 100-episode average reward passed 200. Training typically takes \~10–15 minutes.

<p align="center">
  <img src="images/deep_q_algorithm.png" width="80%">
</p>

---

<a name="10"></a>

## 10 - Watching the Agent in Action

Once trained, I used a helper function to generate a video of the lander successfully reaching the pad using the learned policy.

---

<a name="11"></a>

## References

* [OpenAI Gym: LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)
* [Deep Q-Learning Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602)
* TensorFlow documentation

---

🎯 **Summary**: This project applies Deep Q-Learning with Experience Replay and Target Networks to solve the LunarLander-v2 environment. The agent learns stable control over time and demonstrates the power of reinforcement learning for continuous state-action problems.

```

---
