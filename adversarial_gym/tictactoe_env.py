import gym
from gym import spaces

import numpy as np
from abc import ABC, abstractmethod, abstractproperty
import pickle
import pygame
from . import adversarial

class TicTacToeActionSpace(adversarial.AdversarialActionSpace):

    def __init__(self, env):
        self.env = env
        
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

  
class TicTacToeEnv(adversarial.AdversarialEnv):
    """Abstract TicTacToe Environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, render_size=512, size=3):
        "Set up initial board configuration."

        self.player_X = 1
        self.player_O = -1
        self.draw = 0
        
        self.size = size
        self.render_size = render_size
        self.reset()

        self.action_space = TicTacToeActionSpace(self)
        self.observation_space = spaces.Tuple(spaces=(
            spaces.Box(low=self.player_O, high=self.player_X, shape=(self.size, self.size), dtype=np.int8),
            spaces.Box(low=np.array([self.player_O]),
                       high=np.array([self.player_X]), dtype=np.int8)
        ))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.clock = None
        self.window = None

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

    @property
    def starting_player(self):
        return self.player_X

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

    def _get_canonical_observation(self):
        """
        Returns:
            canonicalState: returns canonical form of board. The canonical form
                            should be independent of players turn. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return self.board * self.current_player, np.array([self.current_player], dtype=np.int8)

    def _get_info(self):
        return {}

    def _game_result(self):
        """
        Returns:
            winner: returns None when game is not finished else returns int value 
                    for the winning player or draw.
               
        """

        for row in self.board:
            if (row == row[0]).all() and row[0] != 0:
                result = row[0]
                reward = 1
                return result, reward

        for column in self.board.T:
            if  (column == column[0]).all() and column[0] != 0:
                result = column[0]
                reward = 1
                return result, reward

        for diagonal in [np.diag(self.board), np.diag(self.board[:, ::-1])]:
            if (diagonal == diagonal[0]).all() and diagonal[0] != 0:
                result = diagonal[0]
                reward = 1
                return result, reward

        # check that the game is complete. If not return None
        if 0 in self.board:
            return None, 0
        
        return self.draw, 1e-4

    def _do_action(self, action):
        s = self.size
        unraveled_action = np.unravel_index(action, (s, s))
        # Add the piece to the empty square.
        assert self.board[unraveled_action] == 0
        self.board[unraveled_action] = self.current_player
        self._current_player = self.previous_player
    
    def _reset_game(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._current_player = self.player_X

    def _get_frame(self):

        canvas = pygame.Surface((self.render_size, self.render_size))
        BG = (210, 180, 140)
        CR = (255, 204, 203)
        CI = (144, 238, 144)
        LI = (35, 31, 32)
        canvas.fill(BG)
        pix_square_size = (
            self.render_size / self.size
        )  # The size of a single grid square in pixels
        LW = int(pix_square_size / 15)
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                LI,
                (0, pix_square_size * x),
                (self.render_size, pix_square_size * x),
                width=LW,
            )
            pygame.draw.line(
                canvas,
                LI,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.render_size),
                width=LW,
            )
        for x in range(self.size):
            for y in range(self.size):
                piece = self.board[x, y]
                center = (pix_square_size * (0.5 + x), pix_square_size * (0.5 + y))
                if piece == self.player_X: 
                    pygame.draw.line( 
                        canvas, 
                        CR, 
                        tuple(np.add(center,(pix_square_size/3, pix_square_size/3))), # start
                        tuple(np.add(center,(-pix_square_size/3, -pix_square_size/3))), # end
                        LW 
                    )
                    pygame.draw.line( 
                        canvas, 
                        CR, 
                        tuple(np.add(center,(-pix_square_size/3, pix_square_size/3))), # start
                        tuple(np.add(center,(pix_square_size/3, -pix_square_size/3))), # end
                        LW
                    )
                elif piece == self.player_O: 
                    pygame.draw.circle(
                        canvas,
                        CI,
                        center,
                        pix_square_size / 2.5,
                    )
                    pygame.draw.circle(
                        canvas,
                        BG,
                        center,
                        pix_square_size / 2.5 - LW,
                    )

        return canvas

    def _get_img(self):
        canvas = self._get_frame()
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ) 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()