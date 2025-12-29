Neuro-Symbolic AI Agent for Hybrid Electric Vehicles 

Overview
This project implements a novel Neuro-Symbolic Control Architecture for the Energy Management System (EMS) of Hybrid Electric Vehicles. Moving beyond traditional rule-based controls, this system integrates:
1.  Deep Reinforcement Learning (DRL): A PPO agent (via Stable-Baselines3) trained on real-world driving data to optimize the power split between the internal combustion engine and the electric motor.
2.  Large Language Models (LLMs): A local Llama 3 instance (via Ollama) acting as a natural language interface, translating vague driver commands (e.g., "I'm in a hurry") into precise control parameters in real-time.

Key Results
- Fuel Efficiency: Achieved 69.7% reduction in fuel consumption compared to baseline strategies.
- Robustness: Implemented fail-safe mechanisms for non-standard user inputs using structured prompting and JSON validation.
- Architecture: Efficient inference pipeline suitable for edge deployment using quantized LLMs.

Tech Stack
- Language: Python 3.10+
- AI/RL: Stable-Baselines3, Gymnasium, PyTorch
- LLM/NLP: Ollama, Llama 3, Prompt Engineering
- Data: Pandas, NumPy (Custom Data Cleaning & Preprocessing Pipeline)
