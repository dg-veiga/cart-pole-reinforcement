import numpy as np
import pandas as pd

class TrainingData:
    def __init__(self):
        self.current_step = np.array([[]])
        self.current_try = np.array([[]])
        self.training_data = np.array([[]])

    def add_step(self, observation, action):
        self.current_try = np.array([list(observation) + [action]]) \
            if self.current_try.size == 0 else \
                np.concatenate((self.current_try, np.array([list(observation) + [action]])), axis=0)

    def close_try(self, final_reward, try_index: int):
        n_steps = self.current_try.shape[0]

        current_try = np.concatenate((self.current_try, np.array(n_steps*[[final_reward]]), np.array(n_steps*[[try_index]])), axis=1)

        self.training_data = current_try \
            if self.training_data.size == 0 else \
                np.concatenate((self.training_data, current_try), axis=0)

    def get_training_data(self, as_dataframe=False):
        if as_dataframe:
            return pd.DataFrame(
                self.training_data, 
                columns = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity', 'Action', 'Reward', 'Try Index']
                )
        return self.training_data
