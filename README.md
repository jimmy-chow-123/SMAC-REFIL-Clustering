# SMAC-REFIL-Clustering


# Dynamic Team Cooperation

This repository implements a dynamic team cooperation algorithm for multi-agent reinforcement learning. The main idea is to enable agents to form and adapt teams dynamically to achieve optimal performance in a shared environment.

## Key Concepts

- **Dynamic Team Formation:**  
  Agents are allowed to form and reconfigure teams on the fly, adapting to the current state and task requirements.

- **Reinforcement Learning:**  
  Each agent employs a reinforcement learning strategy to learn optimal actions that maximize cumulative rewards. The approach is inspired by Deep Q-Network (DQN) methodologies.

- **Auxiliary Embedding Loss:**  
  Beyond learning action values, agents also learn normalized embedding representations. This auxiliary objective encourages similar representations among cooperating agents, enhancing team coordination.

- **Evaluation and Testing:**  
  The system is evaluated in a dynamic environment where agents interact and accumulate rewards over multiple episodes. Performance metrics include cumulative rewards and convergence curves that visualize learning progress.

## Overview

The approach focuses on integrating individual decision-making with effective team collaboration. By combining standard Q-learning with an auxiliary embedding mechanism, agents not only optimize their own actions but also develop representations that facilitate better teamwork. This dual strategy allows the system to thrive in environments where cooperation is critical.

This framework provides a robust platform for experimenting with dynamic team cooperation in multi-agent settings and can be extended to various applications requiring adaptive teamwork.
