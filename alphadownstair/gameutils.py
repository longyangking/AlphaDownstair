import numpy as np 

class Block:
    def __init__(self, position, velocity, bound):
        self.position = np.array(position)
        self.velocity = velocity
        self.bound = bound

    def get_position(self):
        return np.copy(self.position)
    
    def get_velocity(self):
        return self.velocity

    def update(self):
        x, y = self.position
        x += self.velocity
        self.position = (x, y)

        x_min, x_max = self.bound
        if (x < x_min) or (x > x_max):
            return False
        return True

    def set_height(self, height):
        x, y = self.position
        self.position = x, height

class Target:
    def __init__(self, position, step, bound, size):
        self.position = np.array(position)
        self.step = step
        self.bound = bound
        self.size = size

    def get_position(self):
        return np.copy(self.position)

    def check(self, block):
        x, y = block.get_position()
        x0, y0 = self.position
        if y != y0:
            return False
        if (x < x0 + self.size/2) and (x > x0 - self.size/2):
            return True
        return False

    def action(self, code):
        direction = (0, 0)
        if code == 1:
            direction = (1, 0)
            self.__move(direction)
        elif code == 2:
            direction = (-1, 0)
            self.__move(direction)
        elif code == 3:
            pass    # Down stair, do nothing but slide the stairs
            return True

        return False

    def __move(self, direction):
        x, y = self.position
        direction = np.array(direction)
        x += direction[0]*self.step

        x_min, x_max = self.bound
        if (x >= x_min) and (x <= x_max):
            self.position += direction*self.step

class Stair:
    def __init__(self, height, velocity_max, bound):
        self.height = height
        self.velocity_max = velocity_max
        self.bound = bound

        self.block = None
        self.emit(is_start=True)

    def get_block_position(self):
        return self.block.get_position()

    def set_height(self, height):
        self.height = height
        self.block.set_height(height)

    def emit(self, is_start=False):
        x_min, x_max = self.bound

        v = np.random.random()
        velocity = np.rint(self.velocity_max*v)
        if velocity == 0:
            velocity = 1    # Minimum speed

        if is_start:
            v = np.random.random()
            x = np.rint((x_max-x_min)*v + x_min)
            position =  x, self.height

            v = np.random.random()
            if v > 0.5:
                velocity = -velocity
        else:
            v = np.random.random()
            if v > 0.5:
                position =  x_min, self.height
            else:
                position =  x_max, self.height
                velocity = -velocity

        self.block = Block(position=position, velocity=velocity, bound=self.bound)

    def update(self, target):
        status = self.block.update()
        flag = target.check(self.block)

        if not status:
            self.emit()

        return flag

class GameBoard:
    def __init__(self, state_shape, target_step=2, target_size=1, block_velocity_max=3):
        self.state_shape = state_shape
        self.target_step = target_step
        self.target_size = target_size
        self.block_velocity_max = block_velocity_max

        x_min = 0
        x_max, n_stair, channel = self.state_shape
        
        self.n_stair = n_stair
        self.bound = x_min, x_max
        self.target = None
        self.stairs = list()

        self.score = 0

        self.init()

    def get_score(self):
        return self.score

    def init(self):
        self.stairs.append(None)    # Blank block for first stair
        for i in range(1, self.n_stair):
            stair = Stair(
                height=i, 
                velocity_max=self.block_velocity_max, 
                bound=self.bound)
            self.stairs.append(stair)

        x_min, x_max = self.bound
        self.target = Target(
            position=(np.rint((x_max-x_min)/2), 0), 
            step=self.target_step, 
            bound=self.bound, 
            size=self.target_size)

    def generate_stair(self):
        stairs = list()
        for i in range(1, self.n_stair):
            if self.stairs[i] is not None:
                self.stairs[i].set_height(i-1)
            stairs.append(self.stairs[i])

        stair = Stair(
                height=self.n_stair-1, 
                velocity_max=self.block_velocity_max, 
                bound=self.bound)
        stairs.append(stair)

        self.stairs = stairs

    def get_board(self):
        Nx, Ny, channel = self.state_shape
        x_min, x_max = self.bound

        board = np.zeros((Nx, Ny))

        x, y = self.target.get_position()
        board[int(x), int(y)] = 1

        for i in range(self.n_stair):
            if self.stairs[i] is not None:
                x, y = self.stairs[i].get_block_position()
                if (x >= x_min) and (x < x_max):
                    board[int(x), int(y)] = -1

        return board

    def play(self, code):
        is_down = self.target.action(code)
        if is_down:
            self.generate_stair()
            self.score += 1

        flag = False
        for i in range(self.n_stair):
            if self.stairs[i] is not None:
                flag = flag or self.stairs[i].update(self.target)

        return flag

class DownStair:
    def __init__(self, state_shape, ai=None, verbose=None):
        self.state_shape = state_shape
        self.ai = ai
        self.verbose = verbose

        self.gameboard= GameBoard(
            state_shape=self.state_shape,
            target_step=2, 
            target_size=1, 
            block_velocity_max=4
        )
        self.flag = False

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

    def pressaction(self, code):
        self.flag = self.gameboard.play(code=code)

    def start(self):
        from ui import UI

        self.gameboard.init()

        sizeunit = 20
        area = self.gameboard.get_board()
        ui = UI(pressaction=self.pressaction, area=area, sizeunit=sizeunit)
        ui.start()

        if self.verbose:
            count = 0
        
        while not self.flag:
            if self.verbose:
                count += 1

            ui.setarea(area=self.gameboard.get_board())
        
        ui.gameend(self.gameboard.get_score())

        if self.verbose:
            print("Game end with steps [{0}] and score [{1}].".format(count, self.gameboard.get_score()))

    def pressaction_ai(self, code):
        self.flag = self.gameboard.play(code=code)

    def start_ai(self):
        if self.verbose:
            print("Start to play game by AI model...")

        from ui import Viewer

        self.gameboard.init()

        sizeunit = 20
        area = self.gameboard.get_board()
        self.areas.append(area)
        self.update_states()

        viewer = Viewer(area=area, sizeunit=sizeunit)
        viewer.start()
        if self.verbose:
            count = 0

        flag = False
        while not flag:
            if self.verbose:
                count += 1
            
            state = self.get_state()
            action_code = self.ai.play(state)
            flag = self.gameboard.play(code=action_code)
            viewer.setarea(area=self.gameboard.get_board())
        
        viewer.gameend(self.gameengine.get_score())

        if self.verbose:
            print("Game end with steps [{0}] and score [{1}].".format(count, self.gameboard.get_score()))