import collections
import heapq
import random

MAZE = [
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
]

START = (0, 0)
GOAL = (9, 9)

def get_neighbors(pos):
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 10 and 0 <= nc < 10 and MAZE[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def simple_reflex_move(current_pos):
    neighbors = get_neighbors(current_pos)
    return random.choice(neighbors) if neighbors else current_pos

class ModelBasedAgent:
    def __init__(self):
        self.visited = []

    def move(self, current_pos):
        self.visited.append(current_pos)
        neighbors = get_neighbors(current_pos)
        unvisited = [n for n in neighbors if n not in self.visited]
        if unvisited:
            return random.choice(unvisited)
        return random.choice(neighbors)

def goal_based_bfs(start, goal):
    queue = collections.deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == goal:
            return path
        for n in get_neighbors(curr):
            if n not in visited:
                visited.add(n)
                queue.append(path + [n])
    return None

def utility_based_astar(start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, curr = heapq.heappop(frontier)
        if curr == goal: break
        
        for n in get_neighbors(curr):
            new_cost = cost_so_far[curr] + 1
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                priority = new_cost + heuristic(goal, n)
                heapq.heappush(frontier, (priority, n))
                came_from[n] = curr

    path, temp = [], goal
    while temp:
        path.append(temp)
        temp = came_from.get(temp)
    return path[::-1]

def run_simulation():
    print("=== RÉSULTATS DE L'EXERCICE 2 ===\n")

    pos = START
    path_reflex = [pos]
    for _ in range(1000):
        pos = simple_reflex_move(pos)
        path_reflex.append(pos)
        if pos == GOAL: break
    print(f"1. Simple Reflex Agent : {'SUCCÈS' if path_reflex[-1] == GOAL else 'ÉCHEC (Timeout)'}")
    print(f"   Nombre de pas : {len(path_reflex)}")

    agent_model = ModelBasedAgent()
    pos = START
    path_model = [pos]
    for _ in range(1000):
        pos = agent_model.move(pos)
        path_model.append(pos)
        if pos == GOAL: break
    print(f"\n2. Model-based Agent : {'SUCCÈS' if path_model[-1] == GOAL else 'ÉCHEC (Timeout)'}")
    print(f"   Nombre de pas : {len(path_model)}")

    path_goal = goal_based_bfs(START, GOAL)
    print(f"\n3. Goal-based Agent (BFS) : SUCCÈS")
    print(f"   Nombre de pas (optimal) : {len(path_goal)}")
    print(f"   Chemin : {path_goal}")

    path_utility = utility_based_astar(START, GOAL)
    print(f"\n4. Utility-based Agent (A*) : SUCCÈS")
    print(f"   Nombre de pas (optimal) : {len(path_utility)}")
    print(f"   Chemin : {path_utility}")

if __name__ == "__main__":
    run_simulation()