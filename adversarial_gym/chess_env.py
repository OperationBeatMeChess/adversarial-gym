import gym
from gym import spaces
from . import adversarial

import chess
import chess.svg

import numpy as np

from io import BytesIO
import cairosvg
from PIL import Image
import pygame


class ChessActionSpace(adversarial.AdversarialActionSpace):
    def __init__(self, board):
        self.board = board

    @property
    def legal_actions(self):
        return [ChessEnv.move_to_action(move) for move in self.board.legal_moves]
    
    @property
    def action_space_size(self):
        return 64 * 73


class ChessEnv(adversarial.AdversarialEnv):
    """Chess Environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, render_size=512, claim_draw=True, **kwargs):
        self.board = chess.Board(chess960=False)

        self.action_space = ChessActionSpace(self.board)
        self.observation_space = spaces.Tuple(spaces=(
            spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8),
            spaces.Box(low=np.array([False]),
                       high=np.array([True]), dtype=np.bool)
        ))

        self.render_size = render_size
        self.claim_draw = claim_draw

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.clock = None
        self.window = None

    @property
    def current_player(self):
        return self.board.turn

    @property
    def previous_player(self):
        return not self.board.turn

    @property
    def starting_player(self):
        return chess.WHITE

    def get_string_representation(self):
        return self.board.fen()

    def set_string_representation(self, board_string):
        self.board = chess.Board(board_string)
        self.action_space = ChessActionSpace(self.board)

    def _get_canonical_observation(self):
        state = (self.get_piece_configuration(self.board))
        player = self.current_player

        # canonical_representation = - \
        #     state[::-1, ::-1] if player == chess.BLACK else state

        canonical_representation = -state if player == chess.BLACK else state
        canonical_state = canonical_representation, np.array([player], dtype=np.bool)
        return canonical_state

    def _get_info(self):
        info = {
            'castling_rights': self.board.castling_rights,
            'fullmove_number': self.board.fullmove_number,
            'halfmove_clock': self.board.halfmove_clock,
            'promoted': self.board.promoted,
            'ep_square': self.board.ep_square
        }
        return info

    def _do_action(self, action):
        move = self.action_to_move(action)
        self.board.push(move)
    
    def _reset_game(self):
        self.board.reset()

    def _game_result(self):
        result = self.board.result()
        winner = (chess.WHITE if result == '1-0' else chess.BLACK if result ==
                '0-1' else -1 if result == '1/2-1/2' else None)
        # result is 1 for white win or 0 for black win. slight positive for draw
        reward = 0 if result is None else 1e-4 if result == -1 else 1
        return winner, reward

    def _get_frame(self):
        surf = pygame.surfarray.make_surface(self._get_img())
        return pygame.transform.rotate(surf, -90)

    def _get_img(self):
        out = BytesIO()
        bytestring = chess.svg.board(
            self.board, size=self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        image = Image.open(out)
        img = np.asarray(image)
        return img   

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

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