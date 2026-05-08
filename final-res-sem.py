class ReseauSemantique:
    def __init__(self):
        self.noeuds = {}

    def ajouter_relation(self, sujet, relation, objet):
        if sujet not in self.noeuds:
            self.noeuds[sujet] = {}
        if relation not in self.noeuds[sujet]:
            self.noeuds[sujet][relation] = []
        self.noeuds[sujet][relation].append(objet)

    # PARTIE 1 : Propagation de marqueurs
    def propagation_marqueurs(self, depart, relation_cible):
        a_visiter = [depart]
        visites = set()
        resultats = []

        while a_visiter:
            courant = a_visiter.pop(0)
            if courant in visites: continue
            visites.add(courant)

            # Vérification de la relation cible (ex: a-pour-partie)
            if courant in self.noeuds and relation_cible in self.noeuds[courant]:
                resultats.extend(self.noeuds[courant][relation_cible])

            # Remontée du marqueur via les liens Is-a
            if courant in self.noeuds and 'Is-a' in self.noeuds[courant]:
                a_visiter.extend(self.noeuds[courant]['Is-a'])
        
        return list(set(resultats))

    # PARTIE 2 : Héritage (Déduire toutes les propriétés)
    def deduire_tout(self, noeud):
        proprietes = {}
        a_visiter = [noeud]
        
        while a_visiter:
            courant = a_visiter.pop(0)
            if courant in self.noeuds:
                for rel, valeurs in self.noeuds[courant].items():
                    if rel != 'Is-a':
                        if rel not in proprietes:
                            proprietes[rel] = []
                        proprietes[rel].extend(valeurs)
                
                if 'Is-a' in self.noeuds[courant]:
                    a_visiter.extend(self.noeuds[courant]['Is-a'])
        return proprietes

# --- Initialisation avec les données de image_0a574a.png ---
net = ReseauSemantique()

# Relations Is-a
net.ajouter_relation('Plantes', 'Is-a', 'Etres vivants')
net.ajouter_relation('Vertébrés', 'Is-a', 'Etres vivants')
net.ajouter_relation('Oiseaux', 'Is-a', 'Vertébrés')
net.ajouter_relation('Mammifères', 'Is-a', 'Vertébrés')
net.ajouter_relation('Hérons', 'Is-a', 'Oiseaux')
net.ajouter_relation('Moineaux', 'Is-a', 'Oiseaux')
net.ajouter_relation('Chiens', 'Is-a', 'Mammifères')
net.ajouter_relation('Baleine', 'Is-a', 'Mammifères')
net.ajouter_relation('Titi', 'Is-a', 'Moineaux')
net.ajouter_relation('Lassie', 'Is-a', 'Chiens')

# Relations de composition (a-pour-partie)
net.ajouter_relation('Vertébrés', 'a-pour-partie', 'Os')
net.ajouter_relation('Oiseaux', 'a-pour-partie', 'Plumes')

# --- Réponses aux questions du TP ---
print("PARTIE 1 : Propagation de marqueurs")
res_titi = net.propagation_marqueurs('Titi', 'a-pour-partie')
print(f"Question : De quoi est composé Titi ? Réponse : {res_titi}") 

print("\nPARTIE 2 : Héritage et Saturation")
props_lassie = net.deduire_tout('Lassie')
print(f"Propriétés déduites pour Lassie : {props_lassie}")

print("\nPARTIE 3 : Test d'Exception (Inhibition)")
# Si on ajoute une exception : une Baleine n'a pas de pattes (exemple théorique)
net.ajouter_relation('Baleine', 'a-pour-partie', 'Nageoires')
print(f"Propriétés Baleine : {net.deduire_tout('Baleine')}")