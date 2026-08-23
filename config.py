
CONFIG = {

    "state_dim" : 8,
    "action_dim" : 4,
    "hidden_dim" : 128,
    "n_atoms" : 51,

    "learning_rate" : 5e-4,
    "gamma" : 0.99,
    "tau" : 1e-3,

    "epsilon_start" : 1.0,
    "epsilon_end" : 0.01,
    "epsilon_decay" : 0.995,

    "buffer_capacity" : 100000,
    "batch_size" : 64,
    "num_episodes" : 800,
    "num_test_episodes" : 10
    
}


