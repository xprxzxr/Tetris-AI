import random
import cv2
import numpy as np
from PIL import Image
from time import sleep


class Tetris:

    '''NES Tetris (1989) — authentic rules, scoring, colors, and speed system.
    Uses reachability-filtered placement: the AI picks target positions,
    but unreachable placements (due to NES drop speed) are filtered out.'''

    MAP_EMPTY = 0
    # Placed blocks store piece_id + 1 (values 1-7) for colored rendering
    MAP_PLAYER = 8  # Active piece highlight for rendering
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    # ── NES Colors (RGB) ───────────────────────────────────────
    COLORS = {
        0: (0, 0, 0),          # Empty = black
        1: (0, 240, 240),      # I = cyan
        2: (160, 0, 240),      # T = purple
        3: (240, 160, 0),      # L = orange
        4: (0, 0, 240),        # J = blue
        5: (240, 0, 0),        # Z = red
        6: (0, 240, 0),        # S = green
        7: (240, 240, 0),      # O = yellow
        8: (255, 255, 255),    # Active piece = white highlight
    }

    # ── NES Scoring ────────────────────────────────────────────
    # Points = base * (level + 1)
    NES_SCORE_TABLE = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}

    # ── NES Gravity (NTSC frames per drop) ─────────────────────
    NES_GRAVITY = {
        0: 48, 1: 43, 2: 38, 3: 33, 4: 28, 5: 23, 6: 18,
        7: 13, 8: 8, 9: 6, 10: 5, 11: 5, 12: 5, 13: 4,
        14: 4, 15: 4, 16: 3, 17: 3, 18: 3,
    }
    # Levels 19-28: 2 frames, Level 29+: 1 frame (kill screen)

    # ── NES DAS (Delayed Auto Shift) ───────────────────────────
    DAS_INITIAL = 16   # Frames before first horizontal repeat
    DAS_REPEAT = 6     # Frames between subsequent repeats

    SPAWN_X = 3        # NES spawn column

    def __init__(self):
        self.reset()

    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.score = 0
        self.level = 0
        self.total_lines = 0
        self._last_piece = -1  # For NES random

        # NES random: first two pieces
        self.next_piece = self._nes_random_piece()
        # Pre-compute column top positions
        self._col_tops = [Tetris.BOARD_HEIGHT] * Tetris.BOARD_WIDTH
        self._new_round()

        return self._get_board_props(self.board)

    def _nes_random_piece(self):
        '''NES random piece selection: pick 0-7, if 7 or same as last, re-roll to 0-6.'''
        roll = random.randint(0, 7)
        if roll == 7 or roll == self._last_piece:
            roll = random.randint(0, 6)
        self._last_piece = roll
        return roll

    def _get_frames_per_drop(self):
        '''Get NES gravity speed for current level.'''
        if self.level >= 29:
            return 1  # Kill screen
        if self.level >= 19:
            return 2
        return Tetris.NES_GRAVITY.get(self.level, 2)

    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        '''Returns the complete board with NES colors (piece IDs for placed, MAP_PLAYER for active)'''
        piece = self._get_rotated_piece()
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + self.current_pos[1]][x + self.current_pos[0]] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        '''Returns the current game score.'''
        return self.score

    def _new_round(self):
        '''Starts a new round (new piece) — NES random selection'''
        self.current_piece = self.next_piece
        self.next_piece = self._nes_random_piece()
        self.current_pos = [Tetris.SPAWN_X, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] != Tetris.MAP_EMPTY:
                return True
        return False

    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board with NES color (stores piece_id + 1)'''
        board = [x[:] for x in self.board]
        cell_value = self.current_piece + 1  # 1-7 for colored rendering
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = cell_value
        return board

    def _clear_lines(self, board):
        '''Clears completed lines in a board (NES-style: any non-zero cell counts)'''
        lines_to_clear = [i for i, row in enumerate(board) if all(cell != 0 for cell in row)]
        if lines_to_clear:
            board = [row for i, row in enumerate(board) if i not in lines_to_clear]
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _recompute_col_tops(self, board):
        '''Recompute column top positions from a board'''
        col_tops = [Tetris.BOARD_HEIGHT] * Tetris.BOARD_WIDTH
        for col in range(Tetris.BOARD_WIDTH):
            for row in range(Tetris.BOARD_HEIGHT):
                if board[row][col] != Tetris.MAP_EMPTY:
                    col_tops[col] = row
                    break
        return col_tops

    def _get_board_props(self, board):
        '''Get board state — rich features for human-like play.

        Layout (242 floats):
          [0..199]   200  Board binary (1.0 = filled, 0.0 = empty)
          [200..209]  10  Per-column heights (normalized /20)
          [210..219]  10  Per-column hole counts (normalized /20)
          [220..229]  10  Per-column well depth (how much deeper than neighbors, /20)
          [230]        1  Lines cleared this move
          [231]        1  Total holes
          [232]        1  Total bumpiness
          [233]        1  Sum of heights
          [234]        1  Max column height (/20)
          [235]        1  Tetris-readiness (1.0 if a clean well exists for I-piece)
          [236]        1  Row transitions (empty↔filled changes across rows, normalized)
          [237]        1  Column transitions (empty↔filled changes down columns, normalized)
          [238]        1  Level (/29)
          [239]        1  Speed — frames per drop (/48)
          [240]        1  Current piece (/6)
          [241]        1  Next piece (/6)
        '''
        lines, board = self._clear_lines(board)

        # ── Flatten board to 200 binary floats ─────────────────────
        flat = []
        for row in board:
            for cell in row:
                flat.append(1.0 if cell != 0 else 0.0)

        # ── Per-column analysis (single pass) ──────────────────────
        W = Tetris.BOARD_WIDTH
        H = Tetris.BOARD_HEIGHT
        col_heights = [0] * W
        col_holes = [0] * W
        sum_height = 0
        total_holes = 0
        total_bumpiness = 0

        for col in range(W):
            top = H
            for row in range(H):
                if board[row][col] != Tetris.MAP_EMPTY:
                    top = row
                    break

            height = H - top
            col_heights[col] = height
            sum_height += height

            h = 0
            for row in range(top + 1, H):
                if board[row][col] == Tetris.MAP_EMPTY:
                    h += 1
            col_holes[col] = h
            total_holes += h

        for col in range(W - 1):
            total_bumpiness += abs(col_heights[col] - col_heights[col + 1])

        max_height = max(col_heights)

        # ── Well depth per column ──────────────────────────────────
        col_wells = [0] * W
        for col in range(W):
            left_h = col_heights[col - 1] if col > 0 else H
            right_h = col_heights[col + 1] if col < W - 1 else H
            min_neighbor = min(left_h, right_h)
            depth = max(0, min_neighbor - col_heights[col])
            col_wells[col] = depth

        # ── Tetris readiness ───────────────────────────────────────
        tetris_ready = 0.0
        for col in range(W):
            if col_wells[col] >= 4 and col_holes[col] == 0:
                other_heights = [col_heights[c] for c in range(W) if c != col]
                other_bump = sum(abs(other_heights[i] - other_heights[i+1])
                                for i in range(len(other_heights) - 1))
                if other_bump <= 6:
                    tetris_ready = 1.0
                    break

        # ── Row transitions ────────────────────────────────────────
        row_transitions = 0
        for row in range(H):
            for col in range(W - 1):
                a = 1 if board[row][col] != Tetris.MAP_EMPTY else 0
                b = 1 if board[row][col + 1] != Tetris.MAP_EMPTY else 0
                if a != b:
                    row_transitions += 1
            if board[row][0] == Tetris.MAP_EMPTY:
                row_transitions += 1
            if board[row][W - 1] == Tetris.MAP_EMPTY:
                row_transitions += 1

        # ── Column transitions ─────────────────────────────────────
        col_transitions = 0
        for col in range(W):
            for row in range(H - 1):
                a = 1 if board[row][col] != Tetris.MAP_EMPTY else 0
                b = 1 if board[row + 1][col] != Tetris.MAP_EMPTY else 0
                if a != b:
                    col_transitions += 1
            if board[H - 1][col] == Tetris.MAP_EMPTY:
                col_transitions += 1

        # ── Assemble feature vector ────────────────────────────────
        for col in range(W):
            flat.append(col_heights[col] / H)
        for col in range(W):
            flat.append(col_holes[col] / H)
        for col in range(W):
            flat.append(col_wells[col] / H)

        flat.append(float(lines))
        flat.append(total_holes / (W * H))
        flat.append(total_bumpiness / H)
        flat.append(sum_height / (W * H))
        flat.append(max_height / H)
        flat.append(tetris_ready)
        flat.append(row_transitions / (H * (W + 1)))
        flat.append(col_transitions / (W * H))

        flat.append(self.level / 29.0)
        flat.append(self._get_frames_per_drop() / 48.0)
        flat.append(self.current_piece / 6.0)
        flat.append(self.next_piece / 6.0)

        return flat

    def _fast_drop_y(self, piece, pos_x):
        '''Compute drop y-position directly using column tops.'''
        max_y = Tetris.BOARD_HEIGHT
        for px, py in piece:
            col = px + pos_x
            col_top = self._col_tops[col]
            max_y = min(max_y, col_top - py)
        return max_y - 1

    # ── NES Reachability System ────────────────────────────────

    def _can_rotate_to(self, piece_id, pos_x, target_rotation):
        '''Check if piece can rotate from 0 to target_rotation at spawn.
        NES has NO wall kicks — rotation fails on collision.'''
        if target_rotation == 0:
            return True

        rotations_cw = [0, 90, 180, 270]
        cw_steps = rotations_cw.index(target_rotation)
        ccw_steps = (4 - cw_steps) % 4

        # Try CW path (shorter path preferred)
        best_steps = min(cw_steps, ccw_steps)
        if best_steps == cw_steps:
            path = rotations_cw[1:cw_steps + 1]
        else:
            path = [rotations_cw[(-i) % 4] for i in range(1, ccw_steps + 1)]

        for rot in path:
            piece = Tetris.TETROMINOS[piece_id][rot]
            if self._check_collision(piece, [pos_x, 0]):
                return False
        return True

    def _compute_reachable_placements(self):
        '''Return set of (x, rotation) that are physically reachable at current NES speed.'''
        fpd = self._get_frames_per_drop()
        spawn_x = Tetris.SPAWN_X
        piece_id = self.current_piece

        reachable = set()

        if piece_id == 6:
            target_rotations = [0]
        elif piece_id == 0:
            target_rotations = [0, 90]
        else:
            target_rotations = [0, 90, 180, 270]

        for target_rot in target_rotations:
            piece = Tetris.TETROMINOS[piece_id][target_rot]
            min_px = min(p[0] for p in piece)
            max_px = max(p[0] for p in piece)

            # Check rotation is valid at spawn (no wall kicks)
            if not self._can_rotate_to(piece_id, spawn_x, target_rot):
                continue

            # Rotation cost (frames) — take shortest path CW or CCW
            rotations_cw = [0, 90, 180, 270]
            cw_steps = rotations_cw.index(target_rot)
            rot_steps = min(cw_steps, (4 - cw_steps) % 4)
            rot_frames = rot_steps  # 1 frame per rotation press

            # How far piece falls from spawn position
            drop_y = self._fast_drop_y(piece, spawn_x)
            if drop_y < 0:
                continue  # Can't even place at spawn

            total_fall_rows = max(drop_y, 1)
            total_frames = total_fall_rows * fpd

            # Frames available for horizontal movement
            move_frames = max(total_frames - rot_frames, 0)

            # DAS: first move at DAS_INITIAL, then every DAS_REPEAT
            if move_frames < Tetris.DAS_INITIAL:
                max_moves = 0
            else:
                max_moves = 1 + (move_frames - Tetris.DAS_INITIAL) // Tetris.DAS_REPEAT

            # Check each valid x position
            for x in range(-min_px, Tetris.BOARD_WIDTH - max_px):
                moves_needed = abs(x - spawn_x)
                if moves_needed <= max_moves:
                    # Verify path is clear (piece doesn't collide while sliding)
                    if self._path_clear(piece_id, spawn_x, x, target_rot, fpd):
                        reachable.add((x, target_rot))

        return reachable

    def _path_clear(self, piece_id, from_x, to_x, rotation, fpd):
        '''Check the piece can slide horizontally without colliding as gravity pulls it down.'''
        if from_x == to_x:
            return True

        piece = Tetris.TETROMINOS[piece_id][rotation]
        direction = 1 if to_x > from_x else -1
        moves = abs(to_x - from_x)

        current_x = from_x
        for step in range(moves):
            if step == 0:
                frames_elapsed = Tetris.DAS_INITIAL
            else:
                frames_elapsed = Tetris.DAS_INITIAL + step * Tetris.DAS_REPEAT

            rows_dropped = min(frames_elapsed // fpd, Tetris.BOARD_HEIGHT - 1)
            current_x += direction

            if self._check_collision(piece, [current_x, rows_dropped]):
                return False

        return True

    def get_next_states(self):
        '''Get all reachable next states (filtered by NES physics).'''
        reachable = self._compute_reachable_placements()
        states = {}
        piece_id = self.current_piece

        for (x, rotation) in reachable:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            drop_y = self._fast_drop_y(piece, x)

            if drop_y >= 0:
                board = self._add_piece_to_board(piece, [x, drop_y])
                states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_state_size(self):
        '''Size of the state: 200 (board) + 30 (per-col) + 8 (aggregate) + 4 (NES) = 242'''
        return 242

    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation — NES scoring and levels'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # ── Count holes BEFORE placing (for delta penalty) ──────
        holes_before = 0
        for col in range(Tetris.BOARD_WIDTH):
            found_block = False
            for row in range(Tetris.BOARD_HEIGHT):
                if self.board[row][col] != Tetris.MAP_EMPTY:
                    found_block = True
                elif found_block:
                    holes_before += 1

        if render:
            # Animated drop for rendering (speed based on NES level)
            drop_delay = self._get_frames_per_drop() / 60.0  # Convert frames to seconds
            while not self._check_collision(self._get_rotated_piece(), self.current_pos):
                self.render()
                if render_delay:
                    sleep(render_delay)
                else:
                    sleep(max(drop_delay, 0.02))  # Min 20ms for visibility
                self.current_pos[1] += 1
            self.current_pos[1] -= 1
        else:
            # Fast drop (training)
            piece = Tetris.TETROMINOS[self.current_piece][rotation]
            self.current_pos[1] = self._fast_drop_y(piece, x)

        # Place piece on board
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)

        # Update column tops
        self._col_tops = self._recompute_col_tops(self.board)

        # ── Count holes AFTER placing ──────────────────────────
        holes_after = 0
        for col in range(Tetris.BOARD_WIDTH):
            found_block = False
            for row in range(Tetris.BOARD_HEIGHT):
                if self.board[row][col] != Tetris.MAP_EMPTY:
                    found_block = True
                elif found_block:
                    holes_after += 1

        # ── NES Scoring ────────────────────────────────────────
        nes_points = Tetris.NES_SCORE_TABLE.get(lines_cleared, 0) * (self.level + 1)
        self.score += nes_points

        # ── Level advancement (every 10 lines) ─────────────────
        if lines_cleared > 0:
            self.total_lines += lines_cleared
            new_level = self.total_lines // 10
            if new_level > self.level:
                self.level = new_level

        # ── Training reward ─────────────────────────────────────
        reward = 1.0  # Survival bonus

        # NES-proportional line clear reward (Tetris = 30× single)
        if lines_cleared > 0:
            nes_base = Tetris.NES_SCORE_TABLE[lines_cleared]
            reward += nes_base / 10.0  # 4, 10, 30, 120 for 1-4 lines

        # Gentle hole penalty — state features do the heavy lifting
        new_holes = holes_after - holes_before
        if new_holes > 0:
            reward -= new_holes * 0.3

        # Clean board bonus
        if holes_after == 0:
            reward += 0.5

        # Start new round
        self._new_round()
        if self.game_over:
            reward -= 5.0

        return reward, self.game_over

    def render(self):
        '''Renders the current board with NES colors + side panel'''
        board = self._get_complete_board()
        cell_size = 25
        board_w = Tetris.BOARD_WIDTH * cell_size
        board_h = Tetris.BOARD_HEIGHT * cell_size
        panel_w = 160  # Side panel width
        total_w = board_w + panel_w

        # Build board image with NES colors
        img = np.zeros((board_h, total_w, 3), dtype=np.uint8)

        # Draw board cells
        for row in range(Tetris.BOARD_HEIGHT):
            for col in range(Tetris.BOARD_WIDTH):
                cell = board[row][col]
                color = Tetris.COLORS.get(cell, (128, 128, 128))
                y1 = row * cell_size
                x1 = col * cell_size
                img[y1:y1+cell_size, x1:x1+cell_size] = color
                # Grid lines
                if cell == 0:
                    img[y1, x1:x1+cell_size] = (30, 30, 30)
                    img[y1:y1+cell_size, x1] = (30, 30, 30)

        # Draw side panel (dark background)
        img[:, board_w:] = (20, 20, 20)

        # Side panel text (BGR for cv2)
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

        # Next piece preview
        cv2.putText(img, 'Next', (x_text, 230), font, 0.4, white, 1)
        next_piece = Tetris.TETROMINOS[self.next_piece][0]
        next_color = Tetris.COLORS.get(self.next_piece + 1, (255, 255, 255))
        preview_size = 12
        for px, py in next_piece:
            px_draw = board_w + 15 + px * preview_size
            py_draw = 245 + py * preview_size
            img[py_draw:py_draw+preview_size, px_draw:px_draw+preview_size] = next_color

        # Speed info
        fpd = self._get_frames_per_drop()
        cv2.putText(img, f'Speed', (x_text, 310), font, 0.4, white, 1)
        cv2.putText(img, f'{fpd} fpd', (x_text, 330), font, 0.5, cyan, 1)

        if self.level >= 29:
            cv2.putText(img, 'KILL SCREEN', (x_text, 370), font, 0.4, (0, 0, 255), 1)

        # Convert RGB to BGR for cv2
        img = img[..., ::-1]
        cv2.imshow('NES Tetris AI', img)
        cv2.waitKey(1)
