import numpy as np
import random

GRID_SIZE = 4
ACTIONS = [0, 1, 2, 3]
EPISODES = 2000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

class WumpusQLearning:
    def __init__(self):
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        self.start_pos = (0, 0)
        self.gold_pos = (3, 3)
        self.pits = [(1, 1), (2, 2)]
        self.wumpus_pos = (0, 3)

    def reset(self):
        return self.start_pos

    def get_reward(self, state):
        if state == self.gold_pos:
            return 100, True
        if state in self.pits or state == self.wumpus_pos:
            return -100, True
        return -1, False

    def step(self, state, action):
        r, c = state
        if action == 0 and r > 0: r -= 1
        elif action == 1 and r < GRID_SIZE - 1: r += 1
        elif action == 2 and c > 0: c -= 1
        elif action == 3 and c < GRID_SIZE - 1: c += 1
        
        new_state = (r, c)
        reward, done = self.get_reward(new_state)
        return new_state, reward, done

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.choice(ACTIONS)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def train(self):
        print("Entraînement en cours...")
        for i in range(EPISODES):
            state = self.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)
                
                old_value = self.q_table[state[0], state[1], action]
                next_max = np.max(self.q_table[next_state[0], next_state[1]])
                
                new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
                self.q_table[state[0], state[1], action] = new_value
                
                state = next_state
        print("Apprentissage terminé.")

    def play(self):
        state = self.reset()
        path = [state]
        done = False
        while not done:
            action = np.argmax(self.q_table[state[0], state[1]])
            state, _, done = self.step(state, action)
            path.append(state)
            if len(path) > 20: break
        return path

agent = WumpusQLearning()
agent.train()

chemin_final = agent.play()
print(f"Chemin optimal trouvé : {chemin_final}")