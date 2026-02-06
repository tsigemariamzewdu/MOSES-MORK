import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class BetaState:
    """
    Tracks Strength AND Confidence explicitly using Beta Distribution counts.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Evidence for True
        self.beta = beta    # Evidence for False

    @property
    def strength(self):
        # Mean probability: alpha / (alpha + beta)
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self):
        # Evidence saturation. 
        # Map total evidence (counts) to [0, 1] confidence score.
        # "10 units of evidence" = roughly 90% confidence.
        total_evidence = self.alpha + self.beta
        return total_evidence / (total_evidence + 1.0) 

    def __repr__(self):
        return f"Beta(a={self.alpha:.2f}, b={self.beta:.2f})"

class BetaFactorGraph:
    def __init__(self):
        self.nodes = {} 
        self.factors = []

    def get_or_create_node(self, name):
        if name not in self.nodes:
            # Initialize with Neutral Prior (Laplace Smoothing)
            # a=1, b=1 -> Strength=0.5, Conf=Low
            self.nodes[name] = BetaState(1.0, 1.0)
        return self.nodes[name]

    def add_dependency_rule(self, pair_str, rule_strength, rule_confidence):
        """
        Registers a Modus Ponens rule: A -> B
        """
        parts = pair_str.split(' -- ')
        src = parts[0].strip()
        dst = parts[1].strip()
        
        # Ensure nodes exist
        self.get_or_create_node(src)
        self.get_or_create_node(dst)

        # Store the rule params
        self.factors.append({
            'src': src,
            'dst': dst,
            's': rule_strength,
            'c': rule_confidence
        })

    def set_prior(self, name, stv_strength, stv_confidence, base_counts=10.0):
        """
        Anchors a node with external observation (e.g. from Miner).
        This PREVENTS the "floating 0.5" issue.
        """
        node = self.get_or_create_node(name)
        
        # Convert STV (Prob, Conf) -> Beta Counts (Alpha, Beta)
        evidence = stv_confidence * base_counts
        
        # If confidence is 0, we still add epsilon to avoid div/0
        evidence = max(0.1, evidence)
        
        node.alpha = (stv_strength * evidence) + 1.0
        node.beta = ((1.0 - stv_strength) * evidence) + 1.0
    
    def visualize(self, title="Beta Value Network"):
        """
        Visualizes the Factor Graph.
        - Node Color: Strength (Red=0.0, Yellow=0.5, Green=1.0)
        - Node Size: Confidence (Larger = More Confident)
        - Edge Label: Rule Strength & Confidence
        """
        G = nx.DiGraph()
        
        # 1. Add Nodes with attributes
        node_colors = []
        node_sizes = []
        labels = {}
        
        for name, node in self.nodes.items():
            G.add_node(name)
            
            # Color logic: Map Strength 0->1 to Red->Green
            s = node.strength
            node_colors.append(s)
            
            # Size logic: Map Confidence 0->1 to Size 500->3000
            c = node.confidence
            node_sizes.append(1000 + (c * 3000))
            
            # Label: "A\nS:0.9\nC:0.8"
            labels[name] = f"{name}\nS:{s:.2f}\nC:{c:.2f}"

        # 2. Add Edges
        edge_labels = {}
        for rule in self.factors:
            G.add_edge(rule['src'], rule['dst'])
            edge_labels[(rule['src'], rule['dst'])] = f"s:{rule['s']:.2f}\nc:{rule['c']:.2f}"

        # 3. Draw
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=1.5) # Force-directed layout
        
        # Draw Nodes
        nodes = nx.draw_networkx_nodes(G, pos, 
                                     node_color=node_colors, 
                                     node_size=node_sizes, 
                                     cmap=plt.cm.RdYlGn, # Red-Yellow-Green Colormap
                                     vmin=0.0, vmax=1.0,
                                     edgecolors='black')
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)        
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_color='blue')
        
        plt.colorbar(nodes, label="Strength (Probability of True)")
        plt.title(title)
        plt.axis('off')
        # plt.show()
        output_file = "beta_network.png"
        plt.savefig(output_file)
        print(f"Graph visualization saved to: {output_file}")
        plt.close()


    def run_evidence_propagation(self, steps=5):
        print(f"--- Running Beta-Propagation on {len(self.nodes)} Nodes ---")
        
        for i in range(steps):
            max_delta = 0
            
            # Simple Forward Logic Pass
            for rule in self.factors:
                src_node = self.nodes[rule['src']]
                dst_node = self.nodes[rule['dst']]
                
                # 1. Determine Input Strength
                p_src = src_node.strength
                
                # 2. Apply Rule Logic (Weighted Modus Ponens)
                # "If Src is True, Dst is True with prob S"
                # "If Src is False, Dst is... uncertain (0.5)?"
                # Simplified Inference: 
                predicted_strength = (p_src * rule['s']) + ((1.0 - p_src) * 0.5)
                
                # 3. Determine Evidence Flow (The "Valve")
                # We can't be more confident than the Source OR the Rule.
                src_evidence = src_node.alpha + src_node.beta
                rule_capacity = rule['c'] * 20.0 # Scaling factor for rule "stiffness"
                
                # The bottleneck
                evidence_to_pass = min(src_evidence, rule_capacity)
                
                # 4. Convert to Message Counts
                msg_alpha = predicted_strength * evidence_to_pass
                msg_beta = (1.0 - predicted_strength) * evidence_to_pass
                
                # 5. Fuse (Update Destination)
                # Soft update: Average current state with new message
                # (Prevents feedback loops in loopy graphs)
                new_alpha = (dst_node.alpha + msg_alpha) / 2.0
                new_beta = (dst_node.beta + msg_beta) / 2.0
                
                delta = abs(new_alpha - dst_node.alpha)
                max_delta = max(max_delta, delta)
                
                dst_node.alpha = new_alpha
                dst_node.beta = new_beta
            
            print(f"Step {i+1}: Max Evidence Update = {max_delta:.4f}")
            if max_delta < 0.01:
                break

# --- Main Execution ---

data = [
    {'pair': 'A -- B', 'strength': 0.803, 'confidence': 0.3775}, 
    {'pair': 'B -- C', 'strength': 0.749, 'confidence': 0.3039}, 
    {'pair': 'C -- D', 'strength': 0.732, 'confidence': 0.2304},
]

# 1. Init
bg = BetaFactorGraph()

# 2. Load Rules
for row in data:
    bg.add_dependency_rule(row['pair'], row['strength'], row['confidence'])

# 3. CRITICAL STEP: Set Priors (Anchors)
# You MUST tell the graph that at least one node is "True" or "False".
# Otherwise, 0 * 0.8 = 0.
bg.set_prior("A", stv_strength=0.9, stv_confidence=0.8) # "We are pretty sure A is True"

# 4. Run
bg.run_evidence_propagation(steps=10)

# 5. Results
print("\n--- Final STV Results ---")
print(f"{'Variable':<10} | {'Strength':<8} | {'Confidence':<10} | {'Counts (a/b)'}")
for name, node in bg.nodes.items():
    s = node.strength
    c = node.confidence
    print(f"{name:<10} | {s:.4f}   | {c:.4f}     | {node.alpha:.1f}/{node.beta:.1f}")

bg.visualize()