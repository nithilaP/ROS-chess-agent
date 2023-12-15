import chess
import chess.engine
import numpy as np
from edge_detection_utils import convert_to_2d, convert_to_chess_loc

# default_board_dict = {
#     "BLACK_PAWNS": ["a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"],
#     "WHITE_PAWNS": ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"],
#     "BLACK_ROOKS": ["a8", "h8"],
#     "WHITE_ROOKS": ["a1", "h1"],
#     "BLACK_KNIGHTS": ["b8, h8"],
#     "WHITE_KNIGHTS": ["b1", "h1"],
#     "BLACK_BISHOPS": ["c8", "f8"],
#     "WHITE_BISHOPS": ["c1", "f1"],
#     "BLACK_QUEEN": ["d8"],
#     "WHITE_QUEEN": ["d1"],
#     "BLACK_KING": ["e8"],
#     "WHITE_KING": ["e1"]
# }

# def get_default_piece_2d():
#     board_pieces = np.empty((8, 8), dtype=object)
#     for piece_type in default_board_dict:
#         for location in default_board_dict[piece_type]:
#             i, j = convert_to_2d(location)
#             board_pieces[i, j] = piece_type
#     return board_pieces

def get_best_move(board, player):
    if player == "white":
        player = chess.WHITE
    else:
        player = chess.BLACK
    engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")
    board.turn = player
    best_move = engine.play(board, chess.engine.Limit(time=0.1))
    engine.quit()
    return best_move.move

def get_player_move(last_state_mask, curr_state_mask, last_board, current_board_color):
    changes = np.bitwise_xor(last_state_mask, curr_state_mask)
    sum_changes = np.sum(changes)
    x, y = np.where(changes == 1)
    locs = np.column_stack((x, y))
    if sum_changes == 2:
        print("2")
        start_loc = None
        end_loc = None
        for idx, loc in enumerate(locs):
            i, j = loc 
            if not curr_state_mask[i, j]:
                start_loc = loc
                end_loc = locs[1 - idx]
        start_chess_loc = convert_to_chess_loc(start_loc[0], start_loc[1])
        end_chess_loc = convert_to_chess_loc(end_loc[0], end_loc[1])
        return f"{start_chess_loc}{end_chess_loc}"
    elif sum_changes == 1:
        print("1")
        i, j = locs[0]
        start_chess_loc = convert_to_chess_loc(i, j)
        for possible_move in last_board.legal_moves:
            if possible_move[0:2] == start_chess_loc:
               # check for change in color at possible takes
               end_chess_loc = possible_move[2:4]
               last_piece_color = last_board.piece_at(chess.parse_square(end_chess_loc)).color
               if last_piece_color != current_board_color[i, j]:
                   return f"{start_chess_loc}{end_chess_loc}"
    return None

# Get all possible moves given a place, gamestate, player

# print(get_best_move(board_dict=default_board_dict, player="white"))
