import gym
from gym import spaces

import numpy as np

class MoveSpace:
    def __init__(self, board):
        self.board = board

    def sample(self):
        return np.random.choice(list(self.board.legal_moves))
    
class ObmEnv(gym.Env):
    """Chess Environment"""

    def __init__(self, board):
        super(ObmEnv, self).__init__()
        self.board = board

    def _get_image(self):
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size = self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring = bytestring, write_to = out)
        image = Image.open(out)
        return np.asarray(image)

    @staticmethod
    def get_piece_configuration(board):
        piece_map = np.zeros(64)

        for square, piece in zip(board.piece_map().keys(), board.piece_map().values()):
            piece_map[square] = piece.piece_type * (piece.color * 2 - 1)

        return piece_map.reshape((8, 8))

    def _observe(self):
        observation = (self._get_image() if self.observation_mode == 'rgb_array' else self.get_piece_configuration(self.board))
        return observation

    @staticmethod
    def _action_to_move(action):
        from_square = chess.Square(action[0])
        to_square = chess.Square(action[1])
        promotion = (None if action[2] == 0 else chess.Piece(chess.PieceType(action[2])), chess.Color(action[4]))
        drop = (None if action[3] == 0 else chess.Piece(chess.PieceType(action[3])), chess.Color(action[5]))
        move = chess.Move(from_square, to_square, promotion, drop)
        return move
    
    @staticmethod
    def _move_to_action(move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = (0 if move.promotion is None else move.promotion)
        drop = (0 if move.drop is None else move.drop)
        return [from_square, to_square, promotion, drop]

    def step(self, action):
        self.board.push(action)

        observation = self._observe()
        result = self.board.result()
        reward = (1 if result == '1-0' else -1 if result == '0-1' else 0)
        terminal = self.board.is_game_over(claim_draw = self.claim_draw)
        info = {'turn': self.board.turn,
                'castling_rights': self.board.castling_rights,
                'fullmove_number': self.board.fullmove_number,
                'halfmove_clock': self.board.halfmove_clock,
                'promoted': self.board.promoted,
                'chess960': self.board.chess960,
                'ep_square': self.board.ep_square}

        return observation, reward, terminal, info

    def reset(self):
        self.board.reset()

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        return self._observe()

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if not self.viewer is None:
            self.viewer.close()
