import gym
from gym import spaces

import numpy as np
from abc import ABC, abstractmethod, abstractproperty
import pickle

class TicTacToeActionSpace(ABC):

    def __init__(self, env):
        self.env = env

    def sample(self):
        actions = self.legal_actions
        return actions[np.random.randint(len(actions))]
        
    @property
    def legal_actions(self):
        """
        Returns:
            legal_actions: Returns a list of all the legal moves in the current position.
        """
        actions = []

        # Get all the empty squares (color == 0)
        s = self.env.size
        for x in range(s):
            for y in range(s):
                if self.env.board[x][y] == 0:
                    raveled_ind = np.ravel_multi_index((x,y), (s, s))
                    actions.append(raveled_ind)
        return actions
    
    @property
    def action_space_size(self):
        s = self.env.size
        return s * s

  
class TicTacToeEnv(gym.Env):
    """Abstract TicTacToe Environment"""

    def __init__(self, size=3):
        "Set up initial board configuration."

        self.player_X = 1
        self.player_O = -1
        self.draw = 0
        
        self.size = size
        self.reset()

        self.action_space = TicTacToeActionSpace(self)
        self.observation_space = spaces.Tuple(spaces=(
            spaces.Box(low=-1, high=1, shape=(self.size, self.size), dtype=np.int8),
            spaces.Box(low=np.array([False]),
                       high=np.array([True]), dtype=bool)
        ))

    @property
    def current_player(self):
        """
        Returns:
            current_player: Returns identifyier for which player current has their turn.
        """        
        return self._current_player

    @property
    def previous_player(self):
        """
        Returns:
            previous_player: Returns identifyier for which player previously has their turn.
        """
        return -self.current_player

    def get_string_representation(self):
        """
        Returns:
            boardString: Returns string representation of current game state.
        """
        # return self.board.tobytes().hex() + f"#{self.size}"
        return pickle.dumps([self.board, self._current_player, self.size])
    
    def set_string_representation(self, board_string):
        """
        Input:
            boardString: sets game state to match the string representation of board_string.
        """
        # board, size = board_string.split('#')
        # self.size = int(size)
        # self.board = np.frombuffer(bytes.fromhex(board), dtype=self.board.dtype).reshape((self.size, self.size))
        # player = np.sum(self.board)
        # self._current_player = self.player_X if player==0 else self.player_O
        # self.board.setflags(write=True)
        # self.action_space = TicTacToeActionSpace(self)
        self.board, self._current_player, self.size = pickle.loads(board_string)

    def get_canonical_observaion(self):
        """
        Returns:
            canonicalState: returns canonical form of board. The canonical form
                            should be independent of players turn. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return self.board * self.current_player

    def game_result(self):
        """
        Returns:
            winner: returns None when game is not finished else returns int value 
                    for the winning player or draw.
               
        """
        for row in self.board:
            if (row == row[0]).all() and row[0] != 0:
                return row[0]

        for column in self.board.T:
            if  (column == column[0]).all() and column[0] != 0:
                return column[0]

        for diagonal in [np.diag(self.board), np.diag(self.board[:, ::-1])]:
            if (diagonal == diagonal[0]).all() and diagonal[0] != 0:
                return diagonal[0]

        # check that the game is complete. If not return None
        if 0 in self.board:
            return None
        

        return self.draw

    def step(self, action):
        s = self.size
        unraveled_action = np.unravel_index(action, (s, s))
        # Add the piece to the empty square.
        assert self.board[unraveled_action] == 0
        self.board[unraveled_action] = self.current_player
        self._current_player = -self.current_player

        observation = self.get_canonical_observaion()
        result = self.game_result()
        reward = 0 if result is None else 1e-4 if result == 0 else 1
        done = result is not None
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._current_player = self.player_X
        return self.get_canonical_observaion()

    def render(self, mode='human'):

        if mode == 'human':
            print("   ", end="")
            for y in range(self.size):
                print (y,"", end="")
            print("")
            print("  ", end="")
            for _ in range(self.size):
                print ("-", end="-")
            print("--")
            for x in range(self.size):
                print(x, "|",end="")    # print the row #
                for y in range(self.size):
                    piece = self.board[x, y]    # get the piece to print
                    if piece == self.player_X: print("X ",end="")
                    elif piece == self.player_O: print("O ",end="")
                    else:
                        if y==self.size:
                            print("-",end="")
                        else:
                            print("- ",end="")
                print("|")

            print("  ", end="")
            for _ in range(self.size):
                print ("-", end="-")
            print("--")

    def close(self):
        pass