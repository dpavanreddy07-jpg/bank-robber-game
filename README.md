# 🏦 Bank Robber Escape

A single-file Solara game (balanced version) featuring 3 robbers, 2 police officers, 5 money bags, and 1 exit[cite: 1].

## Rules & Mechanics
* **Win Conditions**: 
  * **Robbers win**: The team has $\ge 3$ money bags AND a robber reaches the exit[cite: 1].
  * **Police win**: All robbers are caught or time runs out[cite: 1].
* **AI Pathfinding**: Both sides use A* search with a Manhattan-distance heuristic[cite: 1].
* **Variability**: Results differ from game to game due to randomized money bag and police start positions, random robber bravery, and dynamic decision-making[cite: 1].

## Scenarios
Pick a scenario in the UI to change who is favoured[cite: 1]:
* **Balanced**: Fair fight (roughly 50 / 50)[cite: 1].
* **Robbers win**: Slow, sleepy police far away, only 2 bags needed, no exit guard[cite: 1].
* **Police win**: Police as fast as robbers, start close, 4 bags needed, guard the exit[cite: 1].

## 🚀 Running Locally

1. Install the required dependencies:
   ```bash
   pip install solara
2. Run the application.
   
