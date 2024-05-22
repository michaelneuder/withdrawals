import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

class MarkovQueue:
    def __init__(self, arrival_rate, constraints, max_queue_length):
        self.arrival_rate = arrival_rate
        self.constraints = constraints
        self.max_queue_length = max_queue_length
        self.max_history_length = max(t for _, t in constraints)
        self.min_delta = min(delta for delta, _ in constraints)
        self.state_space = self._create_state_space()
        self.transition_matrix = self._create_transition_matrix()

    def _create_state_space(self):
        # Create all possible states (Q, H)
        queue_lengths = list(range(self.max_queue_length + 1))
        history_elements = list(range(self.min_delta + 1))
        histories = list(product(history_elements, repeat=self.max_history_length))
        state_space = [(q, tuple(h)) for q in queue_lengths for h in histories]
        return state_space

    def _poisson_prob(self, k):
        # Poisson probability mass function
        return np.exp(-self.arrival_rate) * (self.arrival_rate ** k) / np.math.factorial(k)

    def _adjusted_poisson_probabilities(self, queue_length):
        # Calculate the adjusted Poisson probabilities for arrivals
        probabilities = [self._poisson_prob(k) for k in range(self.max_queue_length + 1)]
        total_prob = sum(probabilities)
        
        # Calculate the overflow probability
        overflow_prob = sum(self._poisson_prob(k) for k in range(self.max_queue_length + 1, queue_length + self.max_queue_length + 1))
        
        # Adjust probabilities to ensure they sum to 1
        adjusted_probabilities = [prob / (total_prob + overflow_prob) for prob in probabilities]
        return adjusted_probabilities

    def _minslack(self, queue_length, history):
        max_exits = queue_length
        for delta, t in self.constraints:
            if sum(history[-t:]) > delta:
                return 0
            max_exits = min(max_exits, delta - sum(history[-t:]))
        return max_exits

    def _create_transition_matrix(self):
        state_count = len(self.state_space)
        transition_matrix = np.zeros((state_count, state_count))
        
        for i, (q, h) in enumerate(self.state_space):
            adjusted_probabilities = self._adjusted_poisson_probabilities(q)
            for arrivals in range(self.max_queue_length + 1):
                arrival_prob = adjusted_probabilities[arrivals]
                next_q = min(q + arrivals, self.max_queue_length)
                exits = self._minslack(next_q, h)
                next_state = (next_q - exits, h[1:] + (exits,))
                if next_state in self.state_space:
                    j = self.state_space.index(next_state)
                    transition_matrix[i, j] += arrival_prob
    
        # Ensure each row sums to 1 (handle numerical precision issues)
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = (transition_matrix.T / row_sums).T
                
        return transition_matrix

def find_steady_state(transition_matrix):
    # Compute eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    
    # Find the index of the eigenvalue that is approximately 1
    steady_state_index = np.argmin(np.abs(eigenvalues - 1))
    
    # Get the corresponding eigenvector
    steady_state_vector = np.real(eigenvectors[:, steady_state_index])
    
    # Normalize the eigenvector to make it a probability distribution
    steady_state_vector = steady_state_vector / np.sum(steady_state_vector)
    
    return steady_state_vector

def test_markov_queue():
    constraints = [(2, 2)]
    arrival_rate = .63
    max_queue_length = 50

    queue_system = MarkovQueue(arrival_rate, constraints, max_queue_length)

    num_states = len(queue_system.state_space)
    expected_num_states = (max_queue_length + 1) * (queue_system.min_delta + 1) ** queue_system.max_history_length
    print(f"Expected number of possible states: {expected_num_states}")
    print(f"Actual number of possible states: {num_states}")
    print("Sample states:", queue_system.state_space[:5])

    # Find the steady-state distribution
    steady_state_vector = find_steady_state(queue_system.transition_matrix)
    
    # Calculate the steady-state probabilities for each queue length
    queue_length_probabilities = np.zeros(max_queue_length + 1)
    for (q, _), prob in zip(queue_system.state_space, steady_state_vector):
        queue_length_probabilities[q] += prob

    # Print the steady-state distribution for queue lengths
    print("Steady-state distribution for queue lengths:")
    for q, prob in enumerate(queue_length_probabilities):
        print(f"Queue Length: {q}, Probability: {prob:.4f}")

    # Plot the steady-state distribution for queue lengths
    plt.figure(figsize=(10, 6))
    plt.bar(range(max_queue_length + 1), queue_length_probabilities, color='blue')
    plt.title("Steady-State Distribution for Queue Lengths")
    plt.xlabel("Queue Length")
    plt.ylabel("Probability")
    plt.xticks(range(max_queue_length + 1))
    plt.show()

# Run the test function
test_markov_queue()
