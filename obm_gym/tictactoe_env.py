import gym
from gym import spaces

import numpy as np
from abc import ABC, abstractmethod, abstractproperty

class TicTacToeActionSpace(ABC):

    def __init__(self, board, size=3):
        self.size = size
        self.board = board

    def sample(self):
        actions = self.legal_actions()
        return actions[np.random.randint(len(actions))]

    def legal_actions(self):
        """
        Returns:
            legal_actions: Returns a list of all the legal moves in the current position.
        """
        actions = []

        # Get all the empty squares (color == 0)
        for y in range(self.size):
            for x in range(self.size):
                if self.board[x][y]==0:
                    actions.extend((x,y))
        return actions

  
class TicTacToeEnv(gym.Env):
    """Abstract TicTacToe Environment"""

    def __init__(self, size=3):
        "Set up initial board configuration."

        self.player_X = 1
        self.player_O = -1
        self.draw = 0
        
        self.size = size
        self.board = np.full((self.size, self.size), None)
        self._current_player = self.player_X

        self.action_space = TicTacToeActionSpace(self.board, size=self.size)
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
        return self.board.tostring()+""
    
    def set_string_representation(self, board_string):
        """
        Input:
            boardString: sets game state to match the string representation of board_string.
        """
        self.board = np.fromstring(board_string, dtype=self.board.dtype).reshape(x.shape)
        self.action_space = TicTacToeActionSpace(self.board, size=self.size)

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

        # check that the game is complete. If not return None
        if None in self.board:
            return None
        
        for row in self.board:
            if row.every((v, i, a) => v === a[0]):
                return row[0]

        for column in self.board.T:
            if column.every((v, i, a) => v === a[0]):
                return column[0]

        for diagonal in [np.diag(a), np.diag(a[:, ::-1])]:
            if diagonal.every((v, i, a) => v === a[0]):
                return diagonal[0]

        return self.draw

    def step(self, action):
        (x,y) = move

        # Add the piece to the empty square.
        assert self.board[action] is None
        self.board[action] = self.current_player
        self._current_player = -self._current_player

        observation = self.get_canonical_observaion()
        reward = 0 if self.game_result() is None else 1
        done = result is not None
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.board = np.full((self.size, self.size), None)
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
                print(y, "|",end="")    # print the row #
                for y in range(self.size):
                    piece = self.board[x, y]    # get the piece to print
                    if piece == -1: print("X ",end="")
                    elif piece == 1: print("O ",end="")
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