import numpy as np

class Variable:
    def __init__(self, name):
        self.name = name
        # Belief is [Prob(False), Prob(True)]
        # Initialized to uniform (0.5, 0.5) until we get Unary Factors
        self.belief = np.array([0.5, 0.5])
        self.incoming_messages = {} # Key: Factor, Value: Message Array
        self.neighbors = [] # List of Factor objects

    def __repr__(self):
        return f"Var({self.name})"

class Factor:
    def __init__(self, variables, potential_matrix):
        """
        variables: [Var_Antecedent, Var_Consequent]
        potential_matrix: 2x2 numpy array
        """
        self.variables = variables # [A, B]
        self.potential = potential_matrix
        
        # Register this factor with the variables
        for var in variables:
            var.neighbors.append(self)

    def get_message_to(self, target_var):
        """
        Calculates Sum-Product message from this Factor -> target_var.
        Msg(f->x) = Sum_{y} ( Potential(x,y) * Msg(y->f) )
        """
        # Identify the "other" variable
        if target_var == self.variables[0]:
            other_var = self.variables[1]
            # We are sending to A, summing over B
            # Matrix is [A, B], so we sum over axis 1
            axis_to_sum = 1 
        else:
            other_var = self.variables[0]
            # We are sending to B, summing over A
            axis_to_sum = 0

        # Get the message the 'other' variable sent TO ME (the factor)
        # If no message yet, assume uniform [1, 1]
        incoming = other_var.incoming_messages.get(self, np.array([1.0, 1.0]))

        # CALCULATION:
        # We multiply the potential matrix by the incoming distribution of the other var
        # Then sum out the other var dimension.
        
        # E.g. If summing out col (B), we multiply rows by B's msg? 
        # Actually easier: Element-wise multiply then sum.
        
        # Reshape incoming for broadcasting
        if axis_to_sum == 1: # Summing over B (cols)
            # Incoming is for B (size 2). 
            weighted_potential = self.potential * incoming # Broadcasts over rows
            message = np.sum(weighted_potential, axis=1)   # Sum columns -> Result size 2 (for A)
        else: # Summing over A (rows)
            # Incoming is for A (size 2). 
            # Need to reshape incoming to (2,1) to multiply rows
            weighted_potential = self.potential * incoming[:, np.newaxis]
            message = np.sum(weighted_potential, axis=0)   # Sum rows -> Result size 2 (for B)

        # Normalize to prevent underflow
        return message / np.sum(message)

class FactorGraph:
    def __init__(self):
        self.node_registry = {} # {'A': Variable('A'), 'B': ...}
        self.factors = []

    def get_or_create_node(self, name):
        if name not in self.node_registry:
            self.node_registry[name] = Variable(name)
        return self.node_registry[name]

    def add_dependency_factor(self, pair_str, strength, confidence):
        """
        Parses 'A -- B' and adds a Modus Ponens factor.
        """
        # 1. Parse Nodes
        parts = pair_str.split(' -- ')
        left_name, right_name = parts[0], parts[1]
        
        var_a = self.get_or_create_node(left_name)
        var_b = self.get_or_create_node(right_name)

        # 2. Construct Modus Ponens Matrix (Soft Implication)
        # Logic: A -> B (If A is True, B should be True)
        # Violation: A=True, B=False
        
        # P(Violation) should be low if Strength is high.
        # Let's map Strength directly to the probability of valid states.
        
        # Standard Modus Ponens Table:
        # A  B  | Valid?
        # F  F  | Yes (1.0)
        # F  T  | Yes (1.0) - Vacuously
        # T  T  | Yes (1.0)
        # T  F  | NO  (1.0 - Strength) <--- The Penalty
        
        # Applying Confidence: 
        # If Conf is low, the matrix should flatten towards Uniform (all 1s).
        # We blend the "Strict" matrix with a "Flat" matrix based on Confidence.
        
        violation_cost = 1.0 - (strength * confidence) 
        # (We multiply by confidence so that low conf = cost is close to 1.0 = no penalty)

        # Matrix format: Rows=A (F, T), Cols=B (F, T)
        # [[A=F, B=F], [A=F, B=T]]
        # [[A=T, B=F], [A=T, B=T]]
        
        matrix = np.array([
            [1.0, 1.0],            # A=False (Vacuously True)
            [violation_cost, 1.0]  # A=True  (Violation vs Success)
        ])

        # 3. Create Factor connecting UNIQUE existing nodes
        # Deduplication happens here automatically because we pass the object references
        new_factor = Factor([var_a, var_b], matrix)
        self.factors.append(new_factor)

    def run_belief_propagation(self, steps=10):
        print(f"--- Running BP on {len(self.node_registry)} Nodes & {len(self.factors)} Factors ---")
        
        # Simple Flooding Schedule (Update all nodes, then all factors)
        for i in range(steps):
            # 1. Variables send messages to Factors
            # Msg(x->f) = Product of all incoming messages EXCEPT from f
            for var_name, var in self.node_registry.items():
                # For every neighbor factor
                for target_factor in var.neighbors:
                    msg_out = np.array([1.0, 1.0])
                    # Multiply inputs from OTHER factors
                    for inc_factor, inc_msg in var.incoming_messages.items():
                        if inc_factor != target_factor:
                            msg_out *= inc_msg
                    
                    # Store this message in the target factor's inbox? 
                    # For simplicity in this demo, we update the VARIABLE state 
                    # and let the factor read it from the variable in the next step.
                    # In a real implementation, factors need inboxes too.
                    pass 

            # 2. Factors calculate messages and send to Variables
            # (This is the main update step for this simplified loop)
            max_diff = 0
            for f in self.factors:
                for target_var in f.variables:
                    new_msg = f.get_message_to(target_var)
                    
                    # Store in variable's inbox
                    old_msg = target_var.incoming_messages.get(f, np.array([0.,0.]))
                    target_var.incoming_messages[f] = new_msg
                    
                    diff = np.sum(np.abs(new_msg - old_msg))
                    max_diff = max(max_diff, diff)

            # 3. Update Variable Beliefs (Marginals)
            # Belief(x) = Product of ALL incoming messages
            for var in self.node_registry.values():
                total_belief = np.array([1.0, 1.0]) # Start uniform
                for msg in var.incoming_messages.values():
                    total_belief *= msg
                
                # Normalize
                if np.sum(total_belief) > 0:
                    var.belief = total_belief / np.sum(total_belief)

            print(f"Step {i+1}: Max Message Delta = {max_diff:.4f}")
            if max_diff < 1e-4:
                break
    
    def get_final_stv(self):
        """
        Returns a dictionary {var_name: (Strength, Confidence)}
        Strength (s): The final probability P(True).
        Confidence (c): Heuristic based on how definitive the result is.
                        c = 2 * |0.5 - s|  (0 if 0.5, 1 if 0.0 or 1.0)
        """
        stv_results = {}
        for name, var in self.node_registry.items():
            s = var.belief[1] # Probability of True
            
            # Heuristic Confidence: How far is it from uniform uncertainty?
            c = abs(s - 0.5) * 2 
            
            # Alternative: If you want strictly count-based confidence, 
            # you'd need to modify BP to track evidence counts. 
            # For now, "Certainty" is a standard proxy.
            
            stv_results[name] = (round(s, 4), round(c, 4))
        
        return stv_results
    

# --- Main Execution with your Data ---

data = [
    {'pair': '(OR (AND (NOT A)) (AND (NOT B))) -- (OR (AND A) (AND B))', 'strength': 0.946, 'confidence': 0.1373}, 
    {'pair': 'A -- B', 'strength': 0.803, 'confidence': 0.3775}, 
    {'pair': 'A -- C', 'strength': 0.749, 'confidence': 0.3039},
    {'pair': 'B -- C', 'strength': 0.749, 'confidence': 0.3039}, 
    # ... (Add more rows from your data) ...
]

# 1. Initialize Graph
fg = FactorGraph()

# 2. Load Data (Structure Learning)
for row in data:
    fg.add_dependency_factor(row['pair'], row['strength'], row['confidence'])

# 3. Set a Prior / Observation (The "Unary" Factor)
# Let's say we observe A is definitely True.
# node_A = fg.get_or_create_node('A')
# # Simulate a Unary Factor by creating a dummy message from a "source"
# dummy_factor = Factor([node_A], np.eye(2)) 
# node_A.incoming_messages[dummy_factor] = np.array([0.01, 0.99]) # Strong True

# 4. Run Inference
fg.run_belief_propagation(steps=10)
results = fg.get_final_stv()

# 5. Check Results
print("\n--- Final Posterior Beliefs ---")
print(f"{'Variable':<10} | {'P(True)':<8} | {'Strength':<8} | {'Confidence':<10}")
for name, var in fg.node_registry.items():
    p_true = var.belief[1]
    print(f"{name:<10} : P(True) = {p_true:.4f} | Strength = {results[name][0]:<8} | Confidence = {results[name][1]:<10}")