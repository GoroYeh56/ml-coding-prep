import numpy as np

def q_learning_step(Q, state, action, reward, next_state, alpha=0.1, gamma=0.99):
    # Q: Table of shape (num_states, num_actions)
    
    # Temporal Difference (TD) Target
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state, best_next_action]
    
    # Q-value Update
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    
    return Q