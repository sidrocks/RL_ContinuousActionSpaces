# NeuralNav: Autonomous City Navigation with TD3

## 1. Objective
The goal of this project is to train an autonomous vehicle agent to navigate a complex city map, avoid obstacles (walls), and reach a sequence of designated targets. The agent must learn a continuous control policy to steer and accelerate smoothly without human intervention.

## 2. Problems Addressed
*   **Discrete vs. Continuous Control**: Traditional approaches like DQN rely on discrete actions (e.g., "Turn Left", "Go Straight"), which result in jerky, unnatural movement. Real-world driving requires smooth, continuous control over steering and throttle.
*   **Value Overestimation**: Standard Actor-Critic methods (like DDPG) often suffer from overestimating the value of states, leading to suboptimal or unstable policies.
*   **Sparse Rewards**: Navigating through a maze-like environment with only terminal feedback (crash vs. goal) is ineffective. The agent needs a shaped reward function to guide it.
*   **Brittleness**: Basic RL agents often get stuck in loops or fail to explore effectively.

## 3. Solution
We implemented the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm, a state-of-the-art Reinforcement Learning method specifically designed for continuous action spaces.

*   **Continuous Action Space (2D)**: The model outputs two continuous values in the range `[-1, 1]`:
    1.  **Steering**: Maps to the turning angle (e.g., -20° to +20°).
    2.  **Throttle**: Maps to the car's speed (e.g., `MIN_SPEED` to `MAX_SPEED`).
*   **Sensors**: The car uses 7 ray-cast sensors to detect obstacles at different angles, plus distance and orientation to the target.
*   **Environment**: A custom PyQt6-based simulator (`citymap_assignment - v2.py`) provides the physics, rendering, and interactive map setup.

## 4. Key Concepts of TD3 Algorithm
To address the instability of DDPG, TD3 introduces three critical improvements:

1.  **Clipped Double Q-Learning (Twin Critics)**:
    *   TD3 uses **two** Critic networks ($Q_1$ and $Q_2$).
    *   When calculating the target value, it takes the **minimum** of the two estimates: $y = r + \gamma \min(Q_1(s', a'), Q_2(s', a'))$.
    *   This prevents the agent from being "optimistic" about bad actions, significantly reducing overestimation bias.

2.  **Delayed Policy Updates**:
    *   The Actor (Policy) network is updated less frequently than the Critic (e.g., once every 2 steps).
    *   This allows the Critic's value estimates to settle and become more accurate before the Policy tries to optimize against them, leading to more stable training.

3.  **Target Policy Smoothing**:
    *   Random Gaussian noise is added to the *target* action during the Bellman update: $a' = \text{clip}(\mu(s') + \epsilon, a_{low}, a_{high})$.
    *   This acts as a regularizer, training the value function to be smooth around the chosen action, which prevents the policy from exploiting sharp, incorrect peaks in the Q-function.

## 5. How to Run the Application
### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:
```bash
pip install numpy torch PyQt6
```

### Execution
Run the main simulation file from your terminal:
```bash
python "citymap_assignment - v2.py"
```

## 6. Instructions to Run the App
1.  **Launch**: Start the application using the command above.
2.  **Set Start Position**: Left-click anywhere on the map (black/grey area) to place the **Car** (Blue).
3.  **Set Targets**: Left-click on other locations to place **Targets** (Colored circles). You can create a sequence of multiple targets.
4.  **Finalize Setup**: **Right-click** anywhere on the map to finish setup. The app will enter "READY" mode.
5.  **Start Training**:
    *   Click the **▶ START (Space)** button or press the **Spacebar**.
    *   The car will begin in **WARMUP** phase (random actions) for 1000 steps to fill the replay buffer.
    *   After 1000 steps, it switches to **TRAIN** phase, using the TD3 network to make decisions.
6.  **Configuration**:
    *   **Speed**: Adjust `MIN_SPEED` and `MAX_SPEED` constants in the code to change how fast the car moves.
    *   **Sensors**: Adjust `SENSOR_DIST` to change how far the car sees.

## 7. Demo

<img width="1833" height="1109" alt="image" src="https://github.com/user-attachments/assets/70f29807-da5e-462e-b939-a86d697e8008" />



https://youtu.be/-GC0jrgFnMk




