import numpy as np 
from gameutils import GameBoard

class GameEngine:
    def __init__(self, state_shape, player, verbose=False):
        self.state_shape = state_shape
        self.player = player
        self.verbose = verbose

        self.boards = list()
        self.states = list()

    def get_state(self):
        return self.states[-1]

    def update_states(self):
        '''
        update states
        '''
        Nx, Ny, channel = self.state_shape
        state = np.zeros(self.state_shape)
        n_boards = len(self.boards)
        for i in range(self.channel):
            if i+1 <= n_boards:
                state[:,:,-(i+1)] = self.boards[-(i+1)]

        self.states.append(state)

    def update(self):
        