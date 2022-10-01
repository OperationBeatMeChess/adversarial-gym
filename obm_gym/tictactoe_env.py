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

    def _get_canonical_observaion(self):
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

        observation = self._get_canonical_observaion()
        info = self._get_info()

        result = self.game_result()
        reward = 0 if result is None else 1e-4 if result == 0 else 1
        terminated = result is not None

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._current_player = self.player_X

        observation = self._get_canonical_observaion()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        
        return observation, info
        
    def render(self):
        if self.render_mode == "human":
            
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.render_size, self.render_size))

            canvas = self._render_frame()
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return self._get_frame()

    def _render_frame(self):

        canvas = pygame.Surface((self.render_size, self.render_size))
        BG = (210, 180, 140)
        CR = (255, 204, 203)
        CI = (144, 238, 144)
        LI = (35, 31, 32)
        canvas.fill(BG)
        pix_square_size = (
            self.render_size / self.size
        )  # The size of a single grid square in pixels
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                LI,
                (0, pix_square_size * x),
                (self.render_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                LI,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.render_size),
                width=3,
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
                        5 
                    )
                    pygame.draw.line( 
                        canvas, 
                        CR, 
                        tuple(np.add(center,(-pix_square_size/3, pix_square_size/3))), # start
                        tuple(np.add(center,(pix_square_size/3, -pix_square_size/3))), # end
                        5 
                    )
                elif piece == self.player_O: 
                    pygame.draw.circle(
                        canvas,
                        CI,
                        center,
                        pix_square_size / 2.85,
                    )
                    pygame.draw.circle(
                        canvas,
                        BG,
                        center,
                        pix_square_size / 3,
                    )

        return canvas

    def _get_frame(self):
        canvas = self._render_frame()
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ) 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()