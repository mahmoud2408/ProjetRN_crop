from graphviz import Digraph

class SemanticNetwork:
    def __init__(self):
        self.nodes = {}
        self.exceptions = {}

    def add_relation(self, subject, relation, obj, is_exception=False):
        if subject not in self.nodes:
            self.nodes[subject] = {}
        if relation not in self.nodes[subject]:
            self.nodes[subject][relation] = []
        self.nodes[subject][relation].append(obj)
        
        if is_exception:
            self.exceptions[(subject, obj)] = True

    def marker_propagation(self, start_node, target_relation):
        results = set()
        to_visit = [start_node]
        visited = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self.nodes and target_relation in self.nodes[current]:
                for val in self.nodes[current][target_relation]:
                    results.add(val)

            if current in self.nodes and 'is_a' in self.nodes[current]:
                to_visit.extend(self.nodes[current]['is_a'])
        
        return list(results)

    def get_properties(self, node):
        all_props = {}
        
        def infer(current_node):
            if current_node not in self.nodes:
                return
            
            for rel, values in self.nodes[current_node].items():
                if rel == 'is_a':
                    continue
                for val in values:
                    if rel not in all_props:
                        all_props[rel] = val

            if 'is_a' in self.nodes[current_node]:
                for parent in self.nodes[current_node]['is_a']:
                    infer(parent)

        infer(node)
        return all_props

def dessiner_reseau(network):
    dot = Digraph(comment='Reseau Semantique')
    
    for subject, relations in network.nodes.items():
        for rel, objects in relations.items():
            for obj in objects:
                color = "red" if rel == "is_a" else "blue"
                dot.edge(subject, obj, label=rel, color=color)
    
    dot.render('reseau_semantique', format='png', view=True)

net = SemanticNetwork()

net.add_relation('Animal', 'respire', 'Air')
net.add_relation('Oiseau', 'is_a', 'Animal')
net.add_relation('Oiseau', 'vole', 'Oui')
net.add_relation('Canari', 'is_a', 'Oiseau')
net.add_relation('Autruche', 'is_a', 'Oiseau')
net.add_relation('Autruche', 'vole', 'Non', is_exception=True)

print(net.marker_propagation('Canari', 'respire'))
print(net.get_properties('Canari'))
print(net.get_properties('Autruche'))

dessiner_reseau(net)