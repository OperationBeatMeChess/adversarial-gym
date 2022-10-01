import gym
from gym import spaces

import numpy as np
import pygame
from abc import ABC, abstractmethod, abstractproperty

class AdversarialActionSpace(gym.spaces.Space):

    def sample(self):
        actions = self.legal_actions
        return actions[np.random.randint(len(actions))]

    def contains(self, action, is_legal=True):
        is_contained = action in range(self.action_space_size())
        and_legal = action in self.legal_actions if is_legal else True
        return is_contained and and_legal

    @abstractproperty
    def legal_actions(self):
        """
        Returns:
            legal_actions: Returns a list of all the legal moves in the current position.
        """
        pass
    
    @abstractproperty
    def action_space_size(self):
        """
        Returns:
            action_space_size: returns the number of all possible actions.
        """
        pass
    

  
class AdversarialEnv(gym.Env):
    """Abstract Adversarial Environment"""

    skip_next_render = False

    @abstractproperty
    def current_player(self):
        """
        Returns:
            current_player: Returns identifier for which player currently has their turn.
        """
        pass

    @abstractproperty
    def previous_player(self):
        """
        Returns:
            previous_player: Returns identifier for which player previously had their turn.
        """
        pass

    @abstractproperty
    def starting_player(self):
        """
        Returns:
            starting_player: Returns identifier for which player started the game.
        """
        pass

    @abstractmethod
    def get_string_representation(self):
        """
        Returns:
            board_string: Returns string representation of current game state.
        """
        pass
    
    @abstractmethod
    def set_string_representation(self, board_string):
        """
        Input:
            board_string: sets game state to match the string representation of board_string.
        """
        pass

    @abstractmethod
    def _get_canonical_observation(self):
        """
        Returns:
            canonical_state: returns canonical form of board. The canonical form
                            should be independent of players turn. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
            current_player: returns indentifier of which player is the current player in the canonicial state. 
                            This is used to decode the invariant canonical form.
        """
        pass  

    @abstractmethod
    def _game_result(self):
        """
        Returns:
            winner: returns None when game is not finished else returns int value 
                    for the winning player or draw.
            reward: Reward value given the game result. Should not consider the player who won.
               
        """
        pass

    @abstractmethod
    def _do_action(self, action):
        """
        Input:
            action: Execute action from current game state.
        """
        pass
    
    @abstractmethod
    def _reset_game(self):
        """
        Reset the state of the game to the initial state. 
        This includes reseting the current player to the starting player.
        """

    @abstractmethod
    def _get_frame(self):
        """
        Returns:
            frame: returns py_game frame for the current state of the game. 
            This will be used by render to render the frame for human visualization
               
        """
        pass

    @abstractmethod
    def _get_img(self):
        """
        Returns:
            img: returns rgb_array of the image for the current state of the game. 
               
        """
        pass

    def game_result(self):
        return self._game_result()[0]

    def skip_next_human_render(self):
        """
        Skips the next automatic human render in step or reset. 
        Used for rollouts or similar non visualized moves.
        """
        self.skip_next_render = True

    def step(self, action):
        self._do_action(action)
        observation = self._get_canonical_observation()
        info = self._get_info()
        result, reward = self._game_result()
        terminated = result is not None

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_game()
        observation = self._get_canonical_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        
        return observation, info
        
    def render(self):

        if self.render_mode == "human" and not self.skip_next_render:
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.render_size, self.render_size))

            canvas = self._get_frame()
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return self._get_img()

        self.skip_next_render = False
