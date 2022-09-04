import gym
from gym import spaces
from . import adversarial

import chess
import chess.svg

import numpy as np

from io import BytesIO
import cairosvg
from PIL import Image


class ChessActionSpace(adversarial.AdversarialActionSpace):
    def __init__(self, board):
        self.board = board

    def sample(self):
        moves = list(self.board.legal_moves)
        move = np.random.choice(moves)
        return ChessEnv.move_to_action(move)

    @property
    def legal_actions(self):
        return [ChessEnv.move_to_action(move) for move in self.board.legal_moves]
    
    @property
    def action_space_size(self):
        return 64 * 73


class ChessEnv(adversarial.AdversarialEnv):
    """Chess Environment"""
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, render_size=512, observation_mode='piece_map', claim_draw=True, **kwargs):
        super(ChessEnv, self).__init__()
        self.board = chess.Board(chess960=False)

        self.action_space = ChessActionSpace(self.board)
        self.observation_space = spaces.Tuple(spaces=(
            spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8),
            spaces.Box(low=np.array([False]),
                       high=np.array([True]), dtype=bool)
        ))

        self.render_size = render_size
        self.claim_draw = claim_draw
        self.viewer = None

    @property
    def current_player(self):
        return self.board.turn

    @property
    def previous_player(self):
        return not self.board.turn

    def get_string_representation(self):
        return self.board.fen()

    def set_string_representation(self, board_string):
        self.board = chess.Board(board_string)
        self.action_space = ChessActionSpace(self.board)

    def get_canonical_observaion(self):
        state = (self.get_piece_configuration(self.board))
        player = self.current_player

        canonical_representation = - \
            state[::-1, ::-1] if player == chess.BLACK else state
        canonical_state = (canonical_representation, np.array([player]))
        return canonical_state

    def game_result(self):
        result = self.board.result()
        return (chess.WHITE if result == '1-0' else chess.BLACK if result ==
                '0-1' else -1 if result == '1/2-1/2' else None)

    def set_board_state(self, canonicial_state):
        canonicial_representation, player = canonicial_state
        state = -canonicial_representation[::-1, ::-
                                           1] if player == chess.BLACK else canonicial_representation

        piece_map = {}

        for square, piece in enumerate(state.flatten()):
            if piece:
                color = chess.Color(int(np.sign(piece) > 0))
                piece_map[chess.Square(square)] = chess.Piece(
                    chess.PieceType(abs(piece)), color)

        self.board.set_piece_map(piece_map)

    def step(self, action):
        move = self.action_to_move(action)
        self.board.push(move)

        observation = self.get_canonical_observaion()

        result = self.game_result()
        # result is 1 for white win or 0 for black win. slight positive for draw
        reward = 0 if result is None else 1e-4 if result == -1 else 1
        done = result is not None
        info = {
            'castling_rights': self.board.castling_rights,
            'fullmove_number': self.board.fullmove_number,
            'halfmove_clock': self.board.halfmove_clock,
            'promoted': self.board.promoted,
            'ep_square': self.board.ep_square
        }

        return observation, reward, done, info

    def reset(self):
        self.board.reset()
        return self.get_canonical_observaion()

    def render(self, mode='human'):
        out = BytesIO()
        bytestring = chess.svg.board(
            self.board, size=self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        image = Image.open(out)
        img = np.asarray(image)

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


    def action_to_move(self, action):
        unraveled_action = np.unravel_index(action, (64, 73))
        from_square = unraveled_action[0]

        if unraveled_action[1] < 64:
            to_square = unraveled_action[1]
            move = self.board.find_move(from_square, to_square)
        else:
            pd = unraveled_action[1] - 64
            unraveled_pd = np.unravel_index(pd, (3, 3))
            promotion = unraveled_pd[0] + 2

            from_file = chess.square_file(from_square)
            to_file = unraveled_pd[1] - 1 + from_file
            from_rank = chess.square_rank(from_square)
            to_rank = 0 if 1 == from_rank else 7
            to_square = chess.square(to_file, to_rank)
            move = self.board.find_move(from_square, to_square, promotion=promotion)
        return move

    @staticmethod
    def move_to_action(move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = (0 if move.promotion is None else move.promotion)

        from_file = chess.square_file(from_square)
        to_file   = chess.square_file(to_square)

        if promotion == 0 or promotion == chess.QUEEN:
            action = (from_square, to_square)
            return np.ravel_multi_index(action, (64, 73))
        else:
            d = to_file - from_file + 1 # in {0, 1, 2}
            p = promotion - 2 # in {0, 1, 2}
            pd = np.ravel_multi_index((p, d), (3, 3))
            action = (from_square, 64 + pd) 
            return np.ravel_multi_index(action, (64, 73))

    @staticmethod
    def get_piece_configuration(board):
        piece_map = np.zeros(64, dtype=np.int8)

        for square, piece in board.piece_map().items():
            piece_map[square] = piece.piece_type * (2 * piece.color - 1)

        return piece_map.reshape((8, 8))