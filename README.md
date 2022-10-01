# adversarial-gym

Adversarial gym hosts a range of adversarial turn based games within the OpenAI gym framework. 
The games currently supported are:

1. Chess
2. TicTacToe

## Installation

Depending on the use case you can install in developer mode or using pypi. 

### Use the package manager [pip](https://pip.pypa.io/en/stable/) to install adversarial_gym
```bash
pip install adversarial-gym
```

### Install from Source
Installation from source can be used to edit the environments.This is useful when developing or if your use case requires changes to the current API
```bash
cd Dir/To/Install/In
git clone git@github.com:OperationBeatMeChess/adversarial-gym.git
cd adversarial-gym
pip install -e .
```

## Usage
```python
import gym
import adversarial_gym

# env = gym.make("Chess-v0", render_mode='human')
env = gym.make("TicTacToe-v0", render_mode='human')
print('reset')
env.reset()
terminal = False
while not terminal:
    action = env.action_space.sample()
    observation, reward, terminal, truncated, info = env.step(action)
env.close()
```
## Adversarial Environment API
Each adversarial api follows the structure of the defined base class. This API has a few small additions to the standard OpenAI gym environment to help with the turn based structure of adversarial games. The basic adversarial API follows the below criteria:
```python
class AdversarialEnv(gym.Env):
    """Abstract Adversarial Environment"""

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
        if self.render_mode == "human":
            
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


```

The major differences between a standard gym environment and the adversarial environment is the adversarial environment keeps track of both the game state and each players state. In other words we must know which player is currently making a move and the state which corresponds with this player. Additionally this must be expressed in the result of the game. 

Additional features which were added for convenience were the ability to hash the environment state with a string representation (useful for representing the game as an action tree where each hashed state can search some position). Also, there are a few private member functions required for step and reset. 

Finally, there are two functions used for rendering the pygame window or getting the rgb_array of state. 

This adversarial environment is then also paired with its corresponding adversarial Action_Space. This is required because most games have a subset of the total moves which are legal dependent on the current state of the game. This means it is non trivial to represent the move space with the vanilla gym spaces. To work around this while staying compliant with OpenAI gym API we created the following action space.

```python
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
```
The action space is assumed to be a value in the set
```math
{1, 2, 3, 4, ..., total_number_actions}
```
This means the action space is linear. however, we will have to decode the action into its corresponding move in which ever game. The legal actions will then just be a mask of which actions in the total set of actions can be played in any position. The action space size is just the `total_number_actions`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)