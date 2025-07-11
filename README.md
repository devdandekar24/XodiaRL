# 🎯 Pocket Tanks RL AI Battle

This project implements a Pocket Tanks-style 2-player artillery game using reinforcement learning (RL). Players (tanks) take turns firing projectiles with varying bullet types, velocities, and angles to hit each other. The environment is custom-built using Gymnasium and visualized using Pygame.

A custom reward system is used to train AI agents using Stable Baselines3's A2C algorithm.

🏆 Runner-up AI Tiebreaker Match at Xodia 2024, PICT College  
🎥 Watch the match here: [https://youtu.be/_R-Vf_7Gp9M](https://youtu.be/_R-Vf_7Gp9M)

---

## 🚀 Features

- Custom environment using Gymnasium
- 7 unique bullet types with different behaviors
- Tank movement with range constraints
- Physics simulation with wind effect
- Custom reward logic per bullet type
- Pygame rendering with trajectory visualization
- Trained using Advantage Actor Critic (A2C) RL algorithm
- Play modes: Single bot vs environment or bot vs bot

---

## 🛠️ Setup Instructions

### 1. Install dependencies

```bash
pip install pygame stable-baselines3 gymnasium[all]
```
### 2. Train the bot

```bash
python train.py

```
### 3. Play the game
```
python game.py
```
---
Bullet types

| ID | Name            |
| -- | --------------- |
| 0  | Standard Shell  |
| 1  | Triple Threat   |
| 2  | Long Shot       |
| 3  | Blast Radius    |
| 4  | Healing Halo    |
| 5  | Heavy Impact    |
| 6  | Boomerang Blast |

---
🤖 Reinforcement Learning Details

- Framework: Stable-Baselines3
- Algorithm: A2C (Advantage Actor-Critic)
- Policy: MLP (Multi-layer Perceptron)
- Training Steps: 60,000
---

🥈 Achievement

This project was used in a tiebreaker AI vs AI match during Xodia 24 (2024 edition) at PICT College, where I was runnerup

---
## 📢 Special Thanks

- [@Maheshwar098](https://github.com/Maheshwar098) – For guidance, feedback, or inspiration throughout the development process.

---
