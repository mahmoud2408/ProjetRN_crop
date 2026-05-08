import copy
from collections import deque

class Explorer:
    def __init__(self):
        # Configuration de la carte (W: Wumpus, P: Pit, G: Gold/Treasure)
        self._world_map = [
            ['', '', '', ''],
            ['', 'W', 'P', ''],
            ['', '', 'G', ''],
            ['', '', '', ''],
        ]
        self._position = [1, 1]
        self._alive = True
        self._exited = False

    def _coords_to_index(self, loc):
        row, col = loc
        return row - 1, col - 1

    def _check_hazards(self):
        row, col = self._coords_to_index(self._position)
        cell = self._world_map[row][col]
        if 'P' in cell or 'W' in cell:
            self._alive = False
            print(f"Agent encountered hazard at {self._position}. DEAD.")
        if 'G' in cell:
            print(f"Agent found the treasure at {self._position}!")
            self._exited = True
        return self._alive

    def move(self, action):
        directions = ['Up', 'Down', 'Left', 'Right']
        move_vectors = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        if action not in directions:
            raise ValueError(f"Invalid action: {action}")
        if not self._alive:
            print(f"Cannot move. DEAD at {self._position}")
            return False
        if self._exited:
            print(f"Cannot move. Exited at {self._position}")
            return False
        
        idx = directions.index(action)
        move = move_vectors[idx]
        self._position = [
            min(4, max(1, self._position[0] + move[0])),
            min(4, max(1, self._position[1] + move[1]))
        ]
        print(f"Moved {action}. Current position: {self._position}")
        return self._check_hazards()

    def _adjacent_cells(self):
        adj = []
        for dr, dc in [[0, 1], [0, -1], [-1, 0], [1, 0]]:
            r, c = self._position[0] + dr, self._position[1] + dc
            if 1 <= r <= 4 and 1 <= c <= 4: adj.append([r, c])
        return adj

    def perceive(self):
        if not self._alive:
            print(f"Cannot perceive. DEAD at {self._position}")
            return [None, None]
        if self._exited:
            print(f"Cannot perceive. Exited at {self._position}")
            return [None, None]
        
        breeze, stench = False, False
        for r, c in self._adjacent_cells():
            i, j = self._coords_to_index([r, c])
            cell = self._world_map[i][j]
            if 'P' in cell: breeze = True
            if 'W' in cell: stench = True
        return [breeze, stench]

    def current_location(self):
        return self._position

# --------------------------- Knowledge Base and Utilities ---------------------------

knowledge_base = []
actions_taken = []
current_status = [[0]*4 for _ in range(4)]
allowed_moves = [[0,1],[0,-1],[1,0],[-1,0]]
directions = ['Up','Down','Right','Left']
total_calls = 0

def neighbors(loc):
    adj = []
    for dr, dc in [[0, 1], [0, -1], [-1, 0], [1, 0]]:
        r, c = loc[0] + dr, loc[1] + dc
        if 1 <= r <= 4 and 1 <= c <= 4: adj.append([r, c])
    return adj

def is_valid(r, c):
    return 0 <= r < 4 and 0 <= c < 4

def bfs_path(start, goal):
    visited = [[False]*4 for _ in range(4)]
    q = deque()
    parent = {(start[0], start[1]): None}
    q.append((start[0], start[1]))
    visited[start[0]][start[1]] = True
    
    while q:
        r, c = q.popleft()
        if [r, c] == goal: break
        for idx, (dr, dc) in enumerate(allowed_moves):
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and current_status[nr][nc] == 1 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
                parent[(nr, nc)] = ((r, c), directions[idx])
    
    path = []
    node = (goal[0], goal[1])
    if node not in parent:
        return []
    while node != (start[0], start[1]):
        if parent[node] is None: break
        move = parent[node][1]
        path.append(move)
        node = parent[node][0]
    path.reverse()
    return path

# --------------------------- Logic DPLL Functions ---------------------------

def literal_of(expr):
    for clause in expr:
        for literal in clause:
            return literal[0]

def pure_literals(expr):
    symbols = {lit[0] for clause in expr for lit in clause}
    pure_set, vals = {}, {}
    for s in symbols: pure_set[s] = True
    for clause in expr:
        for lit in clause:
            if not pure_set[lit[0]]: continue
            if lit[0] in vals:
                if vals[lit[0]] != lit[1]: pure_set[lit[0]] = False
            else: vals[lit[0]] = lit[1]
    return {(k, vals[k]) for k in pure_set if pure_set[k]}

def unit_clauses(expr):
    units, consistent = [], True
    tracker = {}
    for clause in expr:
        if len(clause) == 1:
            literal = next(iter(clause))
            units.append({literal})
            if literal[0] not in tracker: tracker[literal[0]] = literal[1]
            elif tracker[literal[0]] != literal[1]: consistent = False
    return consistent, units

def dpll(expr):
    global total_calls
    total_calls += 1
    expr = [frozenset(c) for c in expr]
    expr = [c for c in expr if c]

    ps = pure_literals(expr)
    new_expr = []
    for clause in expr:
        satisfied = False
        for pure_lit in ps:
            if pure_lit in clause:
                satisfied = True
                break
        if not satisfied:
            new_clause = frozenset(l for l in clause if (l[0], 1-l[1]) not in ps)
            new_expr.append(new_clause)
    expr = new_expr

    consistent, units = unit_clauses(expr)
    if not consistent: return False
    for u_clause in units:
        literal = next(iter(u_clause))
        expr = [c for c in expr if literal not in c]
        negated_literal = (literal[0], 1-literal[1])
        expr = [frozenset(l for l in c if l != negated_literal) for c in expr]

    if not expr: return True
    if any(not c for c in expr): return False

    l = literal_of(expr)
    branch_true = [frozenset(l_item for l_item in c if l_item != (l, 0)) for c in expr if (l, 1) not in c]
    if dpll(branch_true): return True
    branch_false = [frozenset(l_item for l_item in c if l_item != (l, 1)) for c in expr if (l, 0) not in c]
    return dpll(branch_false)

# --------------------------- Simulation ---------------------------

def simulate(agent):
    stack = [[1,1]]
    current_status[0][0] = 1
    visited = set()
    visited.add((1,1))

    while agent.current_location() != [3,3] and agent._alive:
        current_cell = agent.current_location()
        breeze, stench = agent.perceive()
        knowledge_base.append({(f'B{current_cell[0]}{current_cell[1]}', int(breeze))})
        knowledge_base.append({(f'S{current_cell[0]}{current_cell[1]}', int(stench))})

        for room in neighbors(current_cell):
            room_tuple = (room[0], room[1])
            if room_tuple in visited: continue
            visited.add(room_tuple)

            # Check for Wumpus
            w_literal = (f'W{room[0]}{room[1]}', 1)
            knowledge_base.append({w_literal})
            if dpll(knowledge_base):
                knowledge_base.pop()
                knowledge_base.append({(w_literal[0], 0)})
                if not dpll(knowledge_base):
                    knowledge_base.pop()
                    knowledge_base.append({w_literal})
                    current_status[room[0]-1][room[1]-1] = 2 # Hazard
                else:
                    knowledge_base.pop()
            else:
                knowledge_base.pop()
                knowledge_base.append({(w_literal[0], 0)})

            # Check for Pits
            if current_status[room[0]-1][room[1]-1] != 2:
                p_literal = (f'P{room[0]}{room[1]}', 1)
                knowledge_base.append({p_literal})
                if dpll(knowledge_base):
                    knowledge_base.pop()
                    knowledge_base.append({(p_literal[0], 0)})
                    if not dpll(knowledge_base):
                        knowledge_base.pop()
                        knowledge_base.append({p_literal})
                        current_status[room[0]-1][room[1]-1] = 2 # Hazard
                    else:
                        knowledge_base.pop()
                else:
                    knowledge_base.pop()
                    knowledge_base.append({(p_literal[0], 0)})

            # Safe to move if not marked as hazard
            if current_status[room[0]-1][room[1]-1] != 2 and room not in stack:
                stack.append(room)
                current_status[room[0]-1][room[1]-1] = 1

        if not stack: break
        next_cell = stack.pop()
        path = bfs_path([current_cell[0]-1, current_cell[1]-1], [next_cell[0]-1, next_cell[1]-1])
        for move in path:
            if not agent._alive: break
            agent.move(move)
            actions_taken.append(move)

    print("\nFinal Moves Taken:", actions_taken)
    print("Total DPLL Calls:", total_calls)


if __name__ == '__main__':
    agent = Explorer()
    simulate(agent)