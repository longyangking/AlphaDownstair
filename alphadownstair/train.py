import numpy as np 
from ai import AI
from gameutils import GameBoard

class SelfplayEngine:
    def __init__(self, ai, verbose=False):
        self.ai = ai 
        self.state_shape = ai.get_state_shape()
        self.verbose = verbose

        # Train data
        self.states = list()
        self.boards = list()
        self.actions = list()
        self.scores = list()

    def get_state(self):
        return self.states[-1]

    def update_states(self):
        '''
        Update stored states
        '''
        Nx, Ny, channel = self.state_shape
        state = np.zeros((Nx,Ny,channel))
        n_areas = len(self.areas)
        for i in range(channel):
            if i+1 <= n_areas:
                state[:,:,-(i+1)] = self.areas[-(i+1)]

        self.states.append(state)

    def start(self, epsilon=0.5, gamma=0.9):
        '''
        The main process of self-play training
        '''

        # TODO  It would be the key to define a appropriate action-value function, 
        #       which would be one of the main parts in this code project.

        gameboard= GameBoard(
            state_shape=self.state_shape,
            target_step=2, 
            target_size=1, 
            block_velocity_max=4
        )
        self.boards.append(gameboard.get_board())
        self.update_states()

        flag = False
        while not flag:
            state = gameboard.get_state()
            action_code = self.ai.play(state)
            flag = gameboard.play(code=action_code)

            self.actions.append(action_code)
            self.scores.append(gameboard.get_score())

            if not flag:    # If game terminates, no data is needed to process
                self.boards.append(gameboard.get_board())
                self.update_states()

        eps = 1e-12
        action_values = list()
        N = len(self.scores)
        for i in range(N):
            action_value = self.ai.evaluate_function(self.state[-(i+1)])
            action_code = self.actions[-(i+1)]
            if i == 0:
                action_value[action_code] = 0
            else:
                current_score = self.scores[-(i+1)]
                if i+2 <= N:
                    previous_score = self.scores[-(i+2)]
                action_value[action_code] = current_score - previous_score
            action_value = np.abs(action_value/np.sum(action_value + eps))
            action_values.append(action_value)

        return self.states, action_values

class TrainAI:
    def __init__(self, state_shape, action_dim=4, ai=None, verbose=False):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.verbose = verbose

        if ai is not None:
            self.ai = ai 
        else:
            self.ai = AI(
                state_shape=state_shape,
                action_dim=action_dim,
                verbose=verbose            
            )

        self.losses = list()
    
    def get_losses(self):
        return np.array(self.losses)

    def get_selfplay_data(self, n_rounds):
        states = list()
        action_values = list()

        if self.verbose:
            starttime = time.time()
            print("Start self-play process with rounds [{0}]:".format(n_rounds))

        for i in range(n_rounds):
            if self.verbose:
                print("{0}th self-play round...".format(i+1))

            engine = SelfplayEngine(
                ai=self.ai,
                verbose=self.verbose
            )

            _states, _action_values = engine.start()
            for i in range(len(_action_values)):
                states.append(_states[i])
                action_values.append(_action_values[i])
        
        if self.verbose:
            endtime = time.time()
            print("End of self-play process with data size [{0}] and cost time [{1:.1f}s].".format(
                len(action_values),  (endtime - starttime)))

        states = np.array(states)
        action_values = np.array(action_values)

        return states, action_values

    def update_ai(self, dataset):
        if self.verbose:
            print("Start to update the network of AI model...")

        history = self.ai.train(dataset, epochs=30, batch_size=32)

        if self.verbose:
            print("End of updating with final loss [{0:.4f}]".format(history.history['loss'][-1]))

        return history

    def start(self, filename):
        '''
        Main training process
        '''
        n_epochs = 1000
        n_rounds = 30
        n_checkpoints = 10

        if self.verbose:
            print("Train AI model with epochs: [{0}]".format(n_epochs))
        
        for i in range(n_epochs):
            if self.verbose:
                print("{0}th self-play training process ...".format(i+1))

            dataset = self.get_selfplay_data(n_rounds)

            history = self.update_ai(dataset)
            self.losses.extend(history.history['loss'])

            if self.verbose:
                print("End of training process.")

            if (i+1)%n_checkpoints == 0:
                if self.verbose:
                    print("Checkpoint: Saving AI model with filename [{0}] ...".format(filename),end="")

                self.ai.save_nnet(filename)

                if self.verbose:
                    print("OK!")
        