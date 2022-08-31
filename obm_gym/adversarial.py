import gym
from gym import spaces

import numpy as np
from abc import ABC, abstractmethod, abstractproperty

class AdversarialActionSpace(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractproperty
    def legal_actions(self):
        """
        Returns:
            legal_actions: Returns a list of all the legal moves in the current position.
        """
        pass

  
class AdversarialEnv(gym.Env):
    """Abstract Adversarial Environment"""
    
    @abstractproperty
    def current_player(self):
        """
        Returns:
            current_player: Returns identifyier for which player current has their turn.
        """
        pass

    @abstractproperty
    def previous_player(self):
        """
        Returns:
            previous_player: Returns identifyier for which player previously has their turn.
        """
        pass

    @abstractmethod
    def get_string_representation(self):
        """
        Returns:
            boardString: Returns string representation of current game state.
        """
        pass
    
    @abstractmethod
    def set_string_representation(self, board_string):
        """
        Input:
            boardString: sets game state to match the string representation of board_string.
        """
        pass

    @abstractmethod
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
        pass  

    @abstractmethod
    def game_result(self):
        """
        Returns:
            winner: returns None when game is not finished else returns int value 
                    for the winning player or draw.
               
        """
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode='human'):
        pass

    @abstractmethod
    def close(self):
        pass