import chess
import chess.engine

default_board_dict = {
    "BLACK_PAWNS": ["a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"],
    "WHITE_PAWNS": ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"],
    "BLACK_ROOKS": ["a8", "h8"],
    "WHITE_ROOKS": ["a1", "h1"],
    "BLACK_KNIGHTS": ["b8, h8"],
    "WHITE_KNIGHTS": ["b1", "h1"],
    "BLACK_BISHOPS": ["c8", "f8"],
    "WHITE_BISHOPS": ["c1", "f1"],
    "BLACK_QUEEN": ["d8"],
    "WHITE_QUEEN": ["d1"],
    "BLACK_KING": ["e8"],
    "WHITE_QUEEN": ["e1"]
}
def get_best_move(board_dict, player):
    if player == "white":
        player = chess.WHITE
    else:
        player = chess.BLACK
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    board = generate_chess_board(board_dict)
    board.turn = player
    best_move = engine.play(board, chess.engine.Limit(time=0.1))
    engine.quit()
    return best_move.move

def generate_chess_board(board_dict):
    board = chess.Board()
    for piece in default_board_dict:
        for idx, start_loc in enumerate(default_board_dict[piece]):
            move = start_loc + board_dict[piece][idx]
            board.push_san(move)
    return board
