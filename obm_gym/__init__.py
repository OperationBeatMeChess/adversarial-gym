from gym.envs.registration import register

register(
    id='Chess-v0',
    entry_point='obm_gym.chess_env:ChessEnv'
)
register(
    id='TicTacToe-v0',
    entry_point='obm_gym.tictactoe_env:TicTacToeEnv'
)