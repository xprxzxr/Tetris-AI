import random
import cv2
import numpy as np
from PIL import Image
from time import sleep
from numba import njit


# ═══════════════════════════════════════════════════════════════════
#  Pre-baked numpy data structures for Numba JIT
# ═══════════════════════════════════════════════════════════════════

# TETROMINOS: shape (7, 4, 4, 2) — [piece_id][rot_idx][block][x,y]
# rot_idx: 0=0°, 1=90°, 2=180°, 3=270°
_TETROMINOS_NP = np.zeros((7, 4, 4, 2), dtype=np.int32)

_TETROMINOS_DICT = {
    0: {0: [(0,0),(1,0),(2,0),(3,0)], 1: [(1,0),(1,1),(1,2),(1,3)],
        2: [(3,0),(2,0),(1,0),(0,0)], 3: [(1,3),(1,2),(1,1),(1,0)]},
    1: {0: [(1,0),(0,1),(1,1),(2,1)], 1: [(0,1),(1,2),(1,1),(1,0)],
        2: [(1,2),(2,1),(1,1),(0,1)], 3: [(2,1),(1,0),(1,1),(1,2)]},
    2: {0: [(1,0),(1,1),(1,2),(2,2)], 1: [(0,1),(1,1),(2,1),(2,0)],
        2: [(1,2),(1,1),(1,0),(0,0)], 3: [(2,1),(1,1),(0,1),(0,2)]},
    3: {0: [(1,0),(1,1),(1,2),(0,2)], 1: [(0,1),(1,1),(2,1),(2,2)],
        2: [(1,2),(1,1),(1,0),(2,0)], 3: [(2,1),(1,1),(0,1),(0,0)]},
    4: {0: [(0,0),(1,0),(1,1),(2,1)], 1: [(0,2),(0,1),(1,1),(1,0)],
        2: [(2,1),(1,1),(1,0),(0,0)], 3: [(1,0),(1,1),(0,1),(0,2)]},
    5: {0: [(2,0),(1,0),(1,1),(0,1)], 1: [(0,0),(0,1),(1,1),(1,2)],
        2: [(0,1),(1,1),(1,0),(2,0)], 3: [(1,2),(1,1),(0,1),(0,0)]},
    6: {0: [(1,0),(2,0),(1,1),(2,1)], 1: [(1,0),(2,0),(1,1),(2,1)],
        2: [(1,0),(2,0),(1,1),(2,1)], 3: [(1,0),(2,0),(1,1),(2,1)]},
}

for pid in range(7):
    for ri in range(4):
        blocks = _TETROMINOS_DICT[pid][ri]
        for bi, (bx, by) in enumerate(blocks):
            _TETROMINOS_NP[pid, ri, bi, 0] = bx
            _TETROMINOS_NP[pid, ri, bi, 1] = by

# NES Gravity: shape (30,) — frames_per_drop for levels 0-29
_NES_GRAVITY_NP = np.array([
    48, 43, 38, 33, 28, 23, 18, 13, 8, 6,  # 0-9
    5, 5, 5, 4, 4, 4, 3, 3, 3,             # 10-18
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,          # 19-28
    1,                                       # 29 (kill screen)
], dtype=np.int32)

# NES Score table: index by lines cleared (0-4)
_NES_SCORE_NP = np.array([0, 40, 100, 300, 1200], dtype=np.int32)

# Constants
_W = np.int32(10)
_H = np.int32(20)
_SPAWN_X = np.int32(3)
_DAS_INITIAL = np.int32(16)
_DAS_REPEAT = np.int32(6)


# ═══════════════════════════════════════════════════════════════════
#  Numba JIT functions — all hot paths compiled to native code
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def _check_collision(piece, pos_x, pos_y, board):
    '''Check if piece collides with board or boundaries.'''
    for i in range(4):
        x = piece[i, 0] + pos_x
        y = piece[i, 1] + pos_y
        if x < 0 or x >= _W or y < 0 or y >= _H or board[y, x] != 0:
            return True
    return False


@njit(cache=True)
def _fast_drop_y(piece, pos_x, col_tops):
    '''Compute drop y-position directly using column tops.'''
    max_y = _H
    for i in range(4):
        px = piece[i, 0]
        py = piece[i, 1]
        col = px + pos_x
        col_top = col_tops[col]
        val = col_top - py
        if val < max_y:
            max_y = val
    return max_y - 1


@njit(cache=True)
def _clear_lines(board):
    '''Clear completed lines. Returns (lines_cleared, new_board).'''
    num_clears = 0
    cleared = np.zeros(_H, dtype=np.int32)

    for row in range(_H):
        is_full = True
        for col in range(_W):
            if board[row, col] == 0:
                is_full = False
                break
        if is_full:
            cleared[num_clears] = row
            num_clears += 1

    if num_clears == 0:
        return 0, board.copy()

    new_board = np.zeros((_H, _W), dtype=np.int32)
    write_idx = _H - 1
    for read_idx in range(_H - 1, -1, -1):
        is_cleared = False
        for i in range(num_clears):
            if read_idx == cleared[i]:
                is_cleared = True
                break
        if not is_cleared:
            for col in range(_W):
                new_board[write_idx, col] = board[read_idx, col]
            write_idx -= 1
    return num_clears, new_board


@njit(cache=True)
def _recompute_col_tops(board):
    '''Recompute column top positions from board.'''
    col_tops = np.full(_W, _H, dtype=np.int32)
    for col in range(_W):
        for row in range(_H):
            if board[row, col] != 0:
                col_tops[col] = row
                break
    return col_tops


@njit(cache=True)
def _add_piece_to_board(piece, pos_x, pos_y, board, cell_value):
    '''Place piece on a copy of board. Returns new board.'''
    new_board = board.copy()
    for i in range(4):
        x = piece[i, 0] + pos_x
        y = piece[i, 1] + pos_y
        new_board[y, x] = cell_value
    return new_board


@njit(cache=True)
def _can_rotate_to(piece_id, pos_x, target_rot_idx, tetrominos, board):
    '''Check rotation path from 0 to target. NES has NO wall kicks.'''
    if target_rot_idx == 0:
        return True

    # Shortest path: CW or CCW
    cw_steps = target_rot_idx
    ccw_steps = (4 - target_rot_idx) % 4

    if cw_steps <= ccw_steps:
        for step in range(1, cw_steps + 1):
            piece = tetrominos[piece_id, step]
            if _check_collision(piece, pos_x, 0, board):
                return False
    else:
        for step in range(1, ccw_steps + 1):
            ri = (4 - step) % 4
            piece = tetrominos[piece_id, ri]
            if _check_collision(piece, pos_x, 0, board):
                return False
    return True


@njit(cache=True)
def _path_clear(piece, from_x, to_x, fpd, board):
    '''Check piece can slide horizontally without collision during gravity.'''
    if from_x == to_x:
        return True

    direction = np.int32(1) if to_x > from_x else np.int32(-1)
    moves = abs(to_x - from_x)
    current_x = from_x

    for step in range(moves):
        if step == 0:
            frames_elapsed = _DAS_INITIAL
        else:
            frames_elapsed = _DAS_INITIAL + step * _DAS_REPEAT

        rows_dropped = frames_elapsed // fpd
        if rows_dropped > _H - 1:
            rows_dropped = _H - 1
        current_x += direction

        if _check_collision(piece, current_x, rows_dropped, board):
            return False
    return True


@njit(cache=True)
def _compute_reachable(piece_id, tetrominos, board, col_tops, fpd):
    '''Return reachable placements as (N, 2) array of [x, rot_idx].'''
    # Max possible: 10 positions × 4 rotations = 40
    result = np.zeros((40, 2), dtype=np.int32)
    count = np.int32(0)

    # Determine valid rotations for this piece
    if piece_id == 6:
        num_rots = 1
    elif piece_id == 0:
        num_rots = 2
    else:
        num_rots = 4

    for rot_idx in range(num_rots):
        piece = tetrominos[piece_id, rot_idx]

        # Find min/max x of piece blocks
        min_px = piece[0, 0]
        max_px = piece[0, 0]
        for i in range(1, 4):
            if piece[i, 0] < min_px:
                min_px = piece[i, 0]
            if piece[i, 0] > max_px:
                max_px = piece[i, 0]

        # Check rotation validity at spawn
        if not _can_rotate_to(piece_id, _SPAWN_X, rot_idx, tetrominos, board):
            continue

        # Rotation cost frames
        cw_steps = rot_idx
        ccw_steps = (4 - rot_idx) % 4
        rot_steps = min(cw_steps, ccw_steps)
        rot_frames = rot_steps

        # Drop distance at spawn
        drop_y = _fast_drop_y(piece, _SPAWN_X, col_tops)
        if drop_y < 0:
            continue

        total_fall_rows = drop_y if drop_y > 0 else 1
        total_frames = total_fall_rows * fpd
        move_frames = total_frames - rot_frames
        if move_frames < 0:
            move_frames = 0

        # DAS calculation
        if move_frames < _DAS_INITIAL:
            max_moves = np.int32(0)
        else:
            max_moves = np.int32(1 + (move_frames - _DAS_INITIAL) // _DAS_REPEAT)

        # Check each valid x
        for x in range(-min_px, _W - max_px):
            moves_needed = abs(x - _SPAWN_X)
            if moves_needed <= max_moves:
                if _path_clear(piece, _SPAWN_X, x, fpd, board):
                    result[count, 0] = x
                    result[count, 1] = rot_idx
                    count += 1

    return result[:count]


@njit(cache=True)
def _count_holes(board):
    '''Count total holes in board.'''
    holes = np.int32(0)
    for col in range(_W):
        found_block = False
        for row in range(_H):
            if board[row, col] != 0:
                found_block = True
            elif found_block:
                holes += 1
    return holes


@njit(cache=True)
def _get_board_props(board, lines_cleared, level, current_piece, next_piece, fpd):
    '''Compute 42-float state vector from board — engineered features only.
    No raw board binary — just the signals that matter for Tetris:
    column heights, holes, wells, bumpiness, transitions, etc.'''
    result = np.zeros(42, dtype=np.float64)

    # Per-column analysis
    col_heights = np.zeros(_W, dtype=np.int32)
    col_holes = np.zeros(_W, dtype=np.int32)
    sum_height = np.int32(0)
    total_holes = np.int32(0)

    for col in range(_W):
        top = _H
        for row in range(_H):
            if board[row, col] != 0:
                top = row
                break
        height = _H - top
        col_heights[col] = height
        sum_height += height

        for row in range(top + 1, _H):
            if board[row, col] == 0:
                col_holes[col] += 1
                total_holes += 1

    # Bumpiness
    total_bumpiness = np.int32(0)
    for col in range(_W - 1):
        d = col_heights[col] - col_heights[col + 1]
        if d < 0:
            d = -d
        total_bumpiness += d

    max_height = np.int32(0)
    for col in range(_W):
        if col_heights[col] > max_height:
            max_height = col_heights[col]

    # Well depths
    col_wells = np.zeros(_W, dtype=np.int32)
    for col in range(_W):
        left_h = col_heights[col - 1] if col > 0 else _H
        right_h = col_heights[col + 1] if col < _W - 1 else _H
        min_n = left_h if left_h < right_h else right_h
        depth = min_n - col_heights[col]
        if depth < 0:
            depth = 0
        col_wells[col] = depth

    # Tetris readiness
    tetris_ready = 0.0
    for col in range(_W):
        if col_wells[col] >= 4 and col_holes[col] == 0:
            other_bump = np.int32(0)
            prev_h = np.int32(-1)
            for c in range(_W):
                if c != col:
                    if prev_h >= 0:
                        d = col_heights[c] - prev_h
                        if d < 0:
                            d = -d
                        other_bump += d
                    prev_h = col_heights[c]
            if other_bump <= 6:
                tetris_ready = 1.0
                break

    # Row transitions
    row_trans = np.int32(0)
    for row in range(_H):
        for col in range(_W - 1):
            a = np.int32(1) if board[row, col] != 0 else np.int32(0)
            b = np.int32(1) if board[row, col + 1] != 0 else np.int32(0)
            if a != b:
                row_trans += 1
        if board[row, 0] == 0:
            row_trans += 1
        if board[row, _W - 1] == 0:
            row_trans += 1

    # Column transitions
    col_trans = np.int32(0)
    for col in range(_W):
        for row in range(_H - 1):
            a = np.int32(1) if board[row, col] != 0 else np.int32(0)
            b = np.int32(1) if board[row + 1, col] != 0 else np.int32(0)
            if a != b:
                col_trans += 1
        if board[_H - 1, col] == 0:
            col_trans += 1

    # Assemble all 42 features
    H_f = np.float64(_H)
    W_f = np.float64(_W)

    # [0..9] Column heights (normalized)
    for col in range(_W):
        result[col] = col_heights[col] / H_f
    # [10..19] Column holes (normalized)
    for col in range(_W):
        result[10 + col] = col_holes[col] / H_f
    # [20..29] Column wells (normalized)
    for col in range(_W):
        result[20 + col] = col_wells[col] / H_f

    # [30] Lines cleared by this placement (0-4)
    result[30] = np.float64(lines_cleared)
    # [31] Total holes (normalized)
    result[31] = np.float64(total_holes) / (W_f * H_f)
    # [32] Total bumpiness (normalized)
    result[32] = np.float64(total_bumpiness) / H_f
    # [33] Aggregate height (normalized)
    result[33] = np.float64(sum_height) / (W_f * H_f)
    # [34] Max column height (normalized)
    result[34] = np.float64(max_height) / H_f
    # [35] Tetris ready (binary)
    result[35] = tetris_ready
    # [36] Row transitions (normalized)
    result[36] = np.float64(row_trans) / (H_f * (W_f + 1.0))
    # [37] Column transitions (normalized)
    result[37] = np.float64(col_trans) / (W_f * H_f)
    # [38] Level (normalized)
    result[38] = np.float64(level) / 29.0
    # [39] Frames per drop (normalized)
    result[39] = np.float64(fpd) / 48.0
    # [40] Current piece (normalized)
    result[40] = np.float64(current_piece) / 6.0
    # [41] Next piece (normalized)
    result[41] = np.float64(next_piece) / 6.0

    return result


# ═══════════════════════════════════════════════════════════════════
#  Tetris class — same public API, JIT internals
# ═══════════════════════════════════════════════════════════════════

class Tetris:
    '''NES Tetris (1989) — Numba JIT accelerated.
    Same public API as before. All hot paths run as native compiled code.'''

    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    MAP_EMPTY = 0
    MAP_PLAYER = 8

    # Keep dict versions for render() only
    TETROMINOS = {
        0: {0: [(0,0),(1,0),(2,0),(3,0)], 90: [(1,0),(1,1),(1,2),(1,3)],
            180: [(3,0),(2,0),(1,0),(0,0)], 270: [(1,3),(1,2),(1,1),(1,0)]},
        1: {0: [(1,0),(0,1),(1,1),(2,1)], 90: [(0,1),(1,2),(1,1),(1,0)],
            180: [(1,2),(2,1),(1,1),(0,1)], 270: [(2,1),(1,0),(1,1),(1,2)]},
        2: {0: [(1,0),(1,1),(1,2),(2,2)], 90: [(0,1),(1,1),(2,1),(2,0)],
            180: [(1,2),(1,1),(1,0),(0,0)], 270: [(2,1),(1,1),(0,1),(0,2)]},
        3: {0: [(1,0),(1,1),(1,2),(0,2)], 90: [(0,1),(1,1),(2,1),(2,2)],
            180: [(1,2),(1,1),(1,0),(2,0)], 270: [(2,1),(1,1),(0,1),(0,0)]},
        4: {0: [(0,0),(1,0),(1,1),(2,1)], 90: [(0,2),(0,1),(1,1),(1,0)],
            180: [(2,1),(1,1),(1,0),(0,0)], 270: [(1,0),(1,1),(0,1),(0,2)]},
        5: {0: [(2,0),(1,0),(1,1),(0,1)], 90: [(0,0),(0,1),(1,1),(1,2)],
            180: [(0,1),(1,1),(1,0),(2,0)], 270: [(1,2),(1,1),(0,1),(0,0)]},
        6: {0: [(1,0),(2,0),(1,1),(2,1)], 90: [(1,0),(2,0),(1,1),(2,1)],
            180: [(1,0),(2,0),(1,1),(2,1)], 270: [(1,0),(2,0),(1,1),(2,1)]},
    }

    COLORS = {
        0: (0, 0, 0), 1: (0, 240, 240), 2: (160, 0, 240),
        3: (240, 160, 0), 4: (0, 0, 240), 5: (240, 0, 0),
        6: (0, 240, 0), 7: (240, 240, 0), 8: (255, 255, 255),
    }

    NES_SCORE_TABLE = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}

    def __init__(self):
        self.reset()

    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = np.zeros((_H, _W), dtype=np.int32)
        self.game_over = False
        self.score = 0
        self.level = 0
        self.total_lines = 0
        self._last_piece = -1

        self.next_piece = self._nes_random_piece()
        self._col_tops = np.full(_W, _H, dtype=np.int32)
        self._new_round()

        fpd = self._get_fpd()
        lines_cleared, clean_board = _clear_lines(self.board)
        return _get_board_props(clean_board, lines_cleared, self.level,
                                self.current_piece, self.next_piece, fpd)

    def _nes_random_piece(self):
        roll = random.randint(0, 7)
        if roll == 7 or roll == self._last_piece:
            roll = random.randint(0, 6)
        self._last_piece = roll
        return roll

    def _get_fpd(self):
        '''Get NES gravity (frames per drop) for current level.'''
        level = min(self.level, 29)
        return int(_NES_GRAVITY_NP[level])

    def _new_round(self):
        self.current_piece = self.next_piece
        self.next_piece = self._nes_random_piece()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        piece = _TETROMINOS_NP[self.current_piece, 0]
        if _check_collision(piece, np.int32(3), np.int32(0), self.board):
            self.game_over = True

    def get_game_score(self):
        return self.score

    def get_state_size(self):
        return 42

    def get_next_states(self):
        '''Get all reachable next states (filtered by NES physics). Returns dict.'''
        fpd = np.int32(self._get_fpd())
        piece_id = np.int32(self.current_piece)
        reachable = _compute_reachable(piece_id, _TETROMINOS_NP, self.board,
                                        self._col_tops, fpd)

        states = {}
        cell_value = np.int32(self.current_piece + 1)

        for idx in range(len(reachable)):
            x = int(reachable[idx, 0])
            rot_idx = int(reachable[idx, 1])
            piece = _TETROMINOS_NP[piece_id, rot_idx]
            drop_y = _fast_drop_y(piece, np.int32(x), self._col_tops)

            if drop_y >= 0:
                new_board = _add_piece_to_board(piece, np.int32(x), np.int32(drop_y),
                                                 self.board, cell_value)
                lines_cleared, clean_board = _clear_lines(new_board)
                props = _get_board_props(clean_board, lines_cleared, self.level,
                                          self.current_piece, self.next_piece, fpd)
                # Map rotation index back to degrees for action
                rot_degrees = rot_idx * 90
                states[(x, rot_degrees)] = props

        return states

    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation — NES scoring and levels'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation
        rot_idx = rotation // 90

        # Count holes BEFORE
        holes_before = _count_holes(self.board)

        if render:
            drop_delay = self._get_fpd() / 60.0
            piece_dict = Tetris.TETROMINOS[self.current_piece][rotation]
            while True:
                # Check collision using list-based piece for render path
                collides = False
                for bx, by in piece_dict:
                    rx = bx + self.current_pos[0]
                    ry = by + self.current_pos[1]
                    if rx < 0 or rx >= 10 or ry < 0 or ry >= 20 or self.board[ry, rx] != 0:
                        collides = True
                        break
                if collides:
                    break
                self.render()
                if render_delay:
                    sleep(render_delay)
                else:
                    sleep(max(drop_delay, 0.02))
                self.current_pos[1] += 1
            self.current_pos[1] -= 1
        else:
            piece = _TETROMINOS_NP[self.current_piece, rot_idx]
            self.current_pos[1] = int(_fast_drop_y(piece, np.int32(x), self._col_tops))

        # Place piece
        piece_np = _TETROMINOS_NP[self.current_piece, rot_idx]
        self.board = _add_piece_to_board(piece_np, np.int32(self.current_pos[0]),
                                          np.int32(self.current_pos[1]),
                                          self.board, np.int32(self.current_piece + 1))
        lines_cleared_i32, self.board = _clear_lines(self.board)
        lines_cleared = int(lines_cleared_i32)
        self._col_tops = _recompute_col_tops(self.board)

        # Count holes AFTER
        holes_after = int(_count_holes(self.board))

        # NES Scoring
        nes_points = Tetris.NES_SCORE_TABLE.get(lines_cleared, 0) * (self.level + 1)
        self.score += nes_points

        # Level advancement
        if lines_cleared > 0:
            self.total_lines += lines_cleared
            new_level = self.total_lines // 10
            if new_level > self.level:
                self.level = new_level

        # Training reward
        reward = 1.0
        if lines_cleared > 0:
            nes_base = Tetris.NES_SCORE_TABLE[lines_cleared]
            reward += nes_base / 10.0

        new_holes = holes_after - int(holes_before)
        if new_holes > 0:
            reward -= new_holes * 0.3

        if holes_after == 0:
            reward += 0.5

        self._new_round()
        if self.game_over:
            reward -= 5.0

        return reward, self.game_over

    def render(self):
        '''Renders the current board with NES colors + side panel'''
        # Build complete board with active piece
        board_render = self.board.copy()
        rot_idx = self.current_rotation // 90
        piece = _TETROMINOS_NP[self.current_piece, rot_idx]
        for i in range(4):
            bx = int(piece[i, 0]) + self.current_pos[0]
            by = int(piece[i, 1]) + self.current_pos[1]
            if 0 <= bx < 10 and 0 <= by < 20:
                board_render[by, bx] = 8  # MAP_PLAYER

        cell_size = 25
        board_w = 10 * cell_size
        board_h = 20 * cell_size
        panel_w = 160
        total_w = board_w + panel_w

        img = np.zeros((board_h, total_w, 3), dtype=np.uint8)

        for row in range(20):
            for col in range(10):
                cell = int(board_render[row, col])
                color = Tetris.COLORS.get(cell, (128, 128, 128))
                y1 = row * cell_size
                x1 = col * cell_size
                img[y1:y1+cell_size, x1:x1+cell_size] = color
                if cell == 0:
                    img[y1, x1:x1+cell_size] = (30, 30, 30)
                    img[y1:y1+cell_size, x1] = (30, 30, 30)

        img[:, board_w:] = (20, 20, 20)

        font = cv2.FONT_HERSHEY_SIMPLEX
        x_text = board_w + 10
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        cyan = (255, 255, 0)

        cv2.putText(img, 'NES TETRIS', (x_text, 25), font, 0.5, yellow, 1)
        cv2.putText(img, f'Score', (x_text, 60), font, 0.4, white, 1)
        cv2.putText(img, f'{self.score}', (x_text, 80), font, 0.5, cyan, 1)
        cv2.putText(img, f'Level', (x_text, 115), font, 0.4, white, 1)
        cv2.putText(img, f'{self.level}', (x_text, 135), font, 0.5, cyan, 1)
        cv2.putText(img, f'Lines', (x_text, 170), font, 0.4, white, 1)
        cv2.putText(img, f'{self.total_lines}', (x_text, 190), font, 0.5, cyan, 1)

        cv2.putText(img, 'Next', (x_text, 230), font, 0.4, white, 1)
        next_piece = _TETROMINOS_NP[self.next_piece, 0]
        next_color = Tetris.COLORS.get(self.next_piece + 1, (255, 255, 255))
        preview_size = 12
        for i in range(4):
            px_draw = board_w + 15 + int(next_piece[i, 0]) * preview_size
            py_draw = 245 + int(next_piece[i, 1]) * preview_size
            img[py_draw:py_draw+preview_size, px_draw:px_draw+preview_size] = next_color

        fpd = self._get_fpd()
        cv2.putText(img, f'Speed', (x_text, 310), font, 0.4, white, 1)
        cv2.putText(img, f'{fpd} fpd', (x_text, 330), font, 0.5, cyan, 1)

        if self.level >= 29:
            cv2.putText(img, 'KILL SCREEN', (x_text, 370), font, 0.4, (0, 0, 255), 1)

        img = img[..., ::-1]
        cv2.imshow('NES Tetris AI', img)
        cv2.waitKey(1)
