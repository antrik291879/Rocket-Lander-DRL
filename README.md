## **Rocket-Lander-using-DRL**

## **Overview**

What is reinforcement learning?
<ul>
  <li>Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent takes actions, receives rewards or penalties, and learns a policy (a strategy) to maximize its total reward over time.</li>
  <li>Imagine training a dog — you give it a treat when it does something right, and nothing when it doesn’t. Over time, it learns what actions get rewards.</li>
  <li>In RL, the agent is that “dog,” the environment is the world it interacts with, and rewards are the treats (or penalties)</li>
  <li>RL lets AI learn from trial and error instead of being told what to do — it becomes more autonomous and adaptive.</li>
</ul>

## **Objective**

This project focuses on training an AI agent to land a rocket safely in the ROCKET LANDER environment, similar to how SpaceX lands their reusable rockets. We'll use Deep Reinforcement Learning (DRL) techniques to teach our rocket to make landing decisions on its own.
Our goal is to create an AI that can control a simulated rocket and guide it to a successful landing. The AI will:
<ul>
  <li>Learn through trial and error</li>
  <li>Make decisions about engine thrust and orientation</li>
  <li>Adapt to changing conditions during descent</li>
</ul>

## **Approach**

## 🌌 **Environment Description**

The custom environment simulates a 2D rocket controlled via thrust and torque.  
It provides continuous feedback about position, velocity, and contact status.

| Observation Index | Description |
|--------------------|-------------|
| 0 | Horizontal Position |
| 1 | Vertical Position |
| 2 | Horizontal Velocity |
| 3 | Vertical Velocity |
| 4 | Angle (radians) |
| 5 | Angular Velocity |
| 6 | Left Leg Contact (1/0) |
| 7 | Right Leg Contact (1/0) |

**Termination Conditions:**
- ✅ Landed successfully  
- 💥 Crashed or out of bounds  
- ⛽ Fuel exhausted  

## 🎮 **Action & State Spaces**

| Action | Range | Description |
|---------|--------|-------------|
| Main Engine Throttle | [0.0, 1.0] | Controls vertical lift |
| Side Engine Thrust | [-1.0, 1.0] | Controls lateral movement |
| Nozzle Angle | [-1.0, 1.0] | Adjusts rocket orientation |

The **state space** is continuous (8-dimensional), allowing the policy to observe smooth dynamics of flight.

<img width="1152" height="863" alt="image" src="https://github.com/user-attachments/assets/e47453dc-b80b-4cb1-84cd-4cbed785438d" />


## 🧮 **Reward Function**

The reward function encourages **safe, stable, and fuel-efficient landings**.  

\[
R_t = \frac{(S_t - S_{t-1})}{10} - 0.3 \times (\text{main\_power} + \text{side\_power})
\]

**Terminal Rewards:**
- **+10** → Landed safely  
- **−10** → Crashed / Out of bounds  

**Penalties:**
- Upward velocity → −1  
- Fuel usage → proportional penalty  

This formulation uses a **potential-based shaping function** to stabilize learning and ensure smooth policy convergence.

## ⚙️ **Training & Hyperparameters**

| Parameter | Value | Description |
|------------|--------|-------------|
| `learning_rate` | 0.0001 | Step size for updating network weights |
| `n_steps` | 1024 | Steps collected before each policy update |
| `n_epochs` | 10 | Gradient passes per update |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | Bias–variance trade-off for advantage estimation |
| `clip_range` | 0.2 | PPO clipping threshold |
| `ent_coef` | 0.01 | Encourages exploration |
| `batch_size` | 64 | Samples per gradient step |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping to ensure stability |

## **Graphs**

<img width="1457" height="706" alt="image" src="https://github.com/user-attachments/assets/bcfb343b-eb84-4dff-9a75-7ee587901b70" />


## **DRL**

Why integrate deep learning?

<ul>
  <li>Traditional RL algorithms like monte carlo or TD learning require storing a Q-table.</li>
  <li>The rocket lander environment has a continuous action and state space.</li>
  <li>Neural networks make computation faster and easier.</li>
</ul>

## **Algorithm** 

A Brief Introduction to PPO

<ul>
  <li>It uses two neurak networks - 1) The actor network and 2) The critic network</li>
  <li>The actor or the policy network is used to output the probabilities of various actions from a state.</li>
  <li>The critic or value network estimates the total expected return from the state.</li>
  <li>The above estimated value is backpropagated through the critic network.</li>
  <li>The policy network is updated after clipping these estimated values.</li>
</ul>

How does PPO work in our case?

<ul>
  <li>Actor Network: takes state → outputs action probabilities (Throttle, Side Thrust, Angle).</li>
  <li>Critic Network: takes same state → outputs single scalar value (predicted total reward).</li>
  <li>Each of these networks has 8 neurons (state vector) in their input layer</li>
  <li>The actor network has 3 output neurons (continuous action space outputs)</li>
  <li>The critic netwrok has one neuron i.e., the vakue estimate</li>
  <li>Both network shares similar architecture but separate weights</li>
  <li>Actor selects optimal thrust commands while the critic evaluates them</li>
</ul>

Advantages of implementing using PPO:

<ul>
  <li>A dense and high variance reward function is designed to take into consideration the complex physics of the environment. The critic network makes sure that these noises do not enter the policy network.</li>
  <li>A single nad batch of data may revert all previous learning. Clipping limits ensure that this doesn't happen and guarantee a smoother and more reliable convergence.</li>
  <li>Both the action space and observation space are in terms of 3 dimensional floating point box. Discretization may lead to loss of accuracy. However, neural networks solve this problem by allowing us to incorporate the floating values into out training.</li>
</ul>

## **Results**
<img width="426" height="240" alt="image" src="https://github.com/user-attachments/assets/7f0998e1-5761-4127-b374-75c1ae318f71" />

| Episode | Total Reward        |
|---------|--------------------|
| 0       | 52.50974502551213  |
| 1       | 54.444107096304684 |
| 2       | 56.27822934129467  |
| 3       | 49.977322842024094 |
| 4       | 54.66340194421085  |
| 5       | 53.1162276354242   |
| 6       | 53.95580809406797  |
| 7       | 52.877967533059824 |
| 8       | 51.99888979974516  |
| 9       | 51.11137140035305  |



