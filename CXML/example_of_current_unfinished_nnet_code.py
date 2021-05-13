from collections import namedtuple
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim

N = 9
NN = N * N
W, B, E = 'O', 'X', '.'
EB = E * NN
WDW = 30



def flatten(c):
    return N * c[0] + c[1]


def unflatten(fc):
    return divmod(fc, N)


def is_on_board(c):
    return c[0] % N == c[0] and c[1] % N == c[1]


def get_valid_neighbors(fc):
    x, y = unflatten(fc)
    possible_neighbors = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    return [flatten(n) for n in possible_neighbors if is_on_board(n)]


NB = [get_valid_neighbors(fc) for fc in range(NN)]
assert sorted(NB[0]) == [1, N]
assert sorted(NB[1]) == [0, 2, N + 1]
assert sorted(NB[N + 1]) == [1, N, N + 2, 2 * N + 1]


def find_reached(board, fc):
    color = board[fc]
    chain = set([fc])
    reached = set()
    frontier = [fc]
    while frontier:
        current_fc = frontier.pop()
        chain.add(current_fc)
        for fn in NB[current_fc]:
            if board[fn] == color and not fn in chain:
                frontier.append(fn)
            elif board[fn] != color:
                reached.add(fn)
    return chain, reached


class IllegalMove(Exception): pass


def place_stone(color, board, fc):
    return board[:fc] + color + board[fc + 1:]


def bulk_place_stones(color, board, stones):
    byteboard = bytearray(board, encoding='ascii')
    color = ord(color)
    for stone in stones:
        byteboard[stone] = color
    return byteboard.decode('ascii')


def maybe_capture_stones(board, fc):
    chain, reached = find_reached(board, fc)
    if not any(board[fr] == E for fr in reached):
        board = bulk_place_stones(E, board, chain)
        return board, chain
    else:
        return board, []


def swap_colors(color):
    if color == E:
        return E
    elif color == B:
        return W
    elif color == W:
        return B
    else:
        print("stupid")


def play_move_incomplete(board, fc, color):
    if board[fc] != E:
        raise IllegalMove
    board = place_stone(color, board, fc)
    opp_color = swap_colors(color)
    opp_stones = []
    my_stones = []
    for fn in NB[fc]:
        if board[fn] == color:
            my_stones.append(fn)
        elif board[fn] == opp_color:
            opp_stones.append(fn)
    for fs in opp_stones:
        board, _ = maybe_capture_stones(board, fs)
    for fs in my_stones:
        board, _ = maybe_capture_stones(board, fs)
    return board


def is_koish(board, fc):
    if board[fc] != E:
        return None
    neighbor_colors = {board[fn] for fn in NB[fc]}
    if len(neighbor_colors) == 1 and not E in neighbor_colors:
        return list(neighbor_colors)[0]
    else:
        return None


class Position(namedtuple('Position', ['board', 'ko'])):
    @staticmethod
    def initial_state():
        return Position(board=EB, ko=None)

    def get_board(self):
        return self.board

    def __str__(self):
        import textwrap
        return '\n'.join(textwrap.wrap(self.board, N))

    def play_move(self, color, fc):
        board, ko = self
        if fc == ko or board[fc] != E:
            raise IllegalMove
        possible_ko_color = is_koish(board, fc)
        new_board = place_stone(color, board, fc)
        opp_color = swap_colors(color)
        opp_stones = []
        my_stones = []
        for fn in NB[fc]:
            if new_board[fn] == color:
                my_stones.append(fn)
            elif new_board[fn] == opp_color:
                opp_stones.append(fn)
        opp_captured = 0
        for fs in opp_stones:
            new_board, captured = maybe_capture_stones(board, fs)
            opp_captured += len(captured)
        new_board, captured = maybe_capture_stones(new_board, fc)
        if captured:
            raise IllegalMove
        for fs in my_stones:
            board, _ = maybe_capture_stones(board, fs)
        if opp_captured == 1 and possible_ko_color == opp_color:
            new_ko = list(opp_captured)[0]
        else:
            new_ko = None
        return Position(new_board, new_ko)

    def score(self):
        board = self.board
        while E in board:
            fempty = board.index(E)
            empties, borders = find_reached(board, fempty)
            pbc = board[list(borders)[0]]
            if all(board[fb] == pbc for fb in borders):
                board = bulk_place_stones(pbc, board, empties)
            else:
                board = bulk_place_stones('?', board, empties)
        return board.count(B) - board.count(W)

    def get_liberties(self):
        board = self.board
        liberties = bytearray(NN)
        for color in (W, B):
            while color in board:
                fc = board.index(color)
                stones, borders = find_reached(board, fc)
                n_l = len([fb for fb in borders if board[fb] == E])
                for fs in stones:
                    liberties[fs] = n_l
                board = bulk_place_stones('?', board, stones)
        return list(liberties)


''' 
-valid_moves()
-over()
-score()
-undo_move()
-make_move()
'''


class Game:
    def __init__(self, device=torch.device("cuda")):
        self.p = Position.initial_state()
        self.ph = []
        self.black_turn = True
        self.is_over = False
        self.last_move = NN
        self.empty = hash(str(self.p))
        self.d = device

    def make_move(self, move):
        '''
        just a note on moves:
        moves are 1 to 1 with the flattened positions on the board.
        there is one extra move for passing.
        the moves are enumerated (0, NN+1), with:
         moves 0 to NN-1 corresponding to placing stone on board
         move NN being a pass
        '''
        if move == NN or self.is_over:
            if self.last_move == NN:
                self.is_over = True
            pass
        cl = B if self.black_turn else W
        self.ph.append(copy.deepcopy(self.p))
        self.p.play_move(cl, move)
        self.black_turn = not self.black_turn
        self.last_move = move

    def undo_move(self):
        if self.ph is None:
            raise Exception("idiot be trynna go backward but he cant")
        self.p = self.ph.pop()
        self.black_turn = not self.black_turn

    def valid_moves(self):
        v = []
        for fc in range(NN):
            try:
                self.make_move(fc)
                v.append(fc)
            except IllegalMove:
                continue
        v.append(NN)
        return v

    def over(self):
        self.is_over = self.is_over or len(self.valid_moves()) == 1
        return self.is_over

    def score(self):
        val = self.p.score()
        return 1.0 if val >= 0 and self.black_turn else -1.0

    def print_game(self):
        print(str(self.p))

    def state(self, nf=1, to_hash=False, flip_states=False):
        nf = nf if nf > 1 else 1
        nf = min(WDW, len(self.ph))
        if to_hash:
            hs = ''
            for a in range(-nf, 0, 1):
                if abs(a) <= len(self.ph):
                    hs += str(self.ph[a])
            return hash(hs)
        gsg = []
        board = self.p.get_board()
        for a in range(nf):
            gs = np.zeros((3, N, N), dtype=np.float)
            for i in range(NN):
                x, y = unflatten(i)
                if board[i] == B:
                    gs[a][i][0, x, y] = 1.0
                elif board[i] == W:
                    gs[a][1, x, y] = 1.0
                if (self.black_turn or flip_states) and not (self.black_turn and flip_states):
                    gs[a][2, x, y] = 1.0
            ggs = torch.tensor(gs, dtype=torch.float, device=self.d).unsqueeze(0)
            gsg.append(ggs)
        return gsg

"""
visits = {}
differential = {}
C = 1.5
h(m):
N = visits("total"), Ni = visits(gamestate), V = differential(gamestate) / Ni
h(m) = V + C * log(N) / Ni

0[100:65] -> ahd(h(m)) / |\ -> rec;
ret
1[99:42] -> ahd(h(m)) / |\ -> rec;
ret
......
f[] -> 0 -> rec;
ret
rec: = record(N += 1, visits(gamestate) += 1, differential(gamestate) += score)
NN:
record:

1
1
1
1
1
1
1
1
1
1
1
1
1
1
9
1
1
9
1
9
9
1
1
1
1
1
1
1
9
1
9
9
9
1
1
1
1
9
9
0
== == == == ==
how
it
grabs
game - changing
play
frames:
(assumes values tracked for each play in the window as well as the first frame out of the window)
1.
calculate
dval
for each frame in window:
    dval = abs(value_i + value_i - 1)
2.
grab
maximum and index
of
max
from softmax of

all
dvals in the
window:
dvmax, i = max(softmax(dval))
3.
check if corresponding
frame is already
present in globals.add if not there.
if dvmax, frame_i frame not in globals: globals.append(dvmax, frame_i)

how
the
model
arch
looks:
globals < --------------------
gamestate(single
frame) -> CONV - FLAT - |-> UTW / ACT -> (policy, value) - - |
---------------------------------
gamestate = hash(16): a
{[P(4) N(4) W(4)]}
16 + (4 + 16 * l)
20 < x < 1332
Node(gs): {N, a(M):{P, N, W}}
nodes.get(node).get(M)
Ni = 5
W = 0.6
P = 0.2
N = 15
0.12 + 1.5 * 0.2 * sqrt(ln(15) / Ni
1.
move
set is always
ordered and is deterministic
from current state

frame
s = game.state(nf=WDW, to_hash=True)
"""

def get_default_hparams():
    value = {}
    value["n_id"] = 1
    value["n_layer"] = 1
    value["f_final"] = 1
    policy = {}
    policy["n_layer"] = 1
    policy["n_id"] = 1
    policy["f_final"] = 1
    hparams = {}
    hparams["gf_n"] = 9
    hparams["gf_in"] = 3
    convcoder = {}
    convcoder["n_layer"] = 4
    convcoder["n_id"] = 2
    convcoder["f_base"] = 32
    convcoder["f_scale"] = 2
    trans = {}
    trans["embd"] = convcoder["f_base"] * convcoder["f_scale"] * hparams["gf_n"] * hparams["gf_n"] // 9
    trans["n_pos"] = WDW
    trans["ctx"] = WDW
    trans["n_ctx"] = WDW
    trans["act_max_steps"] = 12
    trans["n_head"] = 12
    hparams["convcoder"] = convcoder
    hparams["value"] = value
    hparams["policy"] = policy
    hparams["trans"] = trans
    return hparams


class MCTS:
    def __init__(self, game):
        self.nodes = {}
        self.C = 1.5
        a_ = {}
        self.nodes[game.empty] = (1, a_)

    def heuristic_value(self, game, s, a):
        N, a_ = self.nodes.get(s, (1, None))
        p = (1e-5, 0, 0)
        Ni, W, P = p if a_ is None else a_.get(a, p)
        return W * (1.0 / Ni) + self.C * P * np.sqrt(N) / (1 + Ni)

    def record(self, s, a, r):
        n = self.nodes.get(s, None)
        if n is None:
            a_ = {}
            a_[a] = (1, r, 1.0)
            self.nodes[s] = (1, a_)
        else:
            N, a_ = self.nodes[s]
            N += 1
            n_a = a_.get(a, (0, 0, 0))
            n_a[0] += 1
            n_a[1] += r
            n_a[2] = n_a[0] / N
            a_[a] = n_a
            self.nodes[s] = (N, a_)

    def playout_value(self, game):
        s = game.state(nf=WDW, to_hash=True)
        if game.over():
            self.record(s, NN, -game.score())
            return -game.score()
        ahd = {}
        for a in game.valid_moves():
            ahd[a] = -self.heuristic_value(game, s, a)
        move = max(ahd, key=ahd.get)
        game.make_move(move)
        value = -self.playout_value(game)
        game.undo_move()
        self.record(s, move, value)
        return value

    def monte_carlo_value(self, game, N=100):
        scores = [self.playout_value(game) for i in range(0, N)]
        return np.mean(scores)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Conv1D(nn.Module):
    def __init__(self, nfeatures, nx):
        super(Conv1D, self).__init__()
        self.nf = nfeatures
        w = torch.empty(nx, nfeatures)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nfeatures))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        print(n_state)
        assert n_state % n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        newx = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*newx)

    def split_heads(self, x, k=False):
        newx = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*newx)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            pkey, pvalue = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((pkey, key), dim=-1)
            value = torch.cat((pvalue, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class Attention2(nn.Module):
    def __init__(self, nx, n_ctx, n_head, scale=False):
        super(Attention2, self).__init__()
        n_state = nx
        assert n_state % n_head == 0
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        newx = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*newx)

    def split_heads(self, x, k=False):
        newx = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*newx)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past):
        query = self.c_attn(x)
        query = self.split_heads(query)

        key, value = layer_past[0].transpose(-2, -1), layer_past[1]
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


class Attention_(nn.Module):
    def __init__(self, nx, n_ctx, n_head, scale=False):
        super(Attention_, self).__init__()
        n_state = nx
        assert n_state % n_head == 0
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.wq = Conv1D(n_state, nx)
        self.wk = Conv1D(n_state, nx)
        self.wv = Conv1D(n_state, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        newx = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*newx)

    def split_heads(self, x, k=False):
        newx = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*newx)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, enc):
        query = self.wq(x)
        query = self.split_heads(query)

        key = self.wk(enc)
        value = self.wv(enc)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        nx = n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class RConv(nn.Module):
    def __init__(self, filtre, feature):
        super(RConv, self).__init__()
        assert (feature % 2 == 1)
        self.ln_1 = nn.BatchNorm2d(filtre)
        self.rl_1 = nn.ReLU()
        self.cn_1 = nn.Conv2d(filtre, filtre, feature, (feature - 1) // 2)
        self.ln_2 = nn.BatchNorm2d(filtre)
        self.rl_2 = nn.ReLU()
        self.cn_2 = nn.Conv2d(filtre, filtre, feature, (feature - 1) // 2)

    def forward(self, x):
        x = self.cn_1(self.rl_1(self.ln_1(x)))
        x = self.cn_2(self.rl_2(self.ln_2(x)))
        return x


"""
n_layer: number of convolutional blocks
f_base: number of filters to use for the first block
f_scale: scaling factor for how to increase the number of filters per block
gf_in: number of channels the input frame for the game has
gf_n: the side length of the square game frame
n_id: number of final 1x1 convolutional blocks
"""


class ConvCoder(nn.Module):
    def __init__(self, hparams):
        super(ConvCoder, self).__init__()
        self.n_layer = hparams["convcoder"]["n_layer"]
        self.f_base = hparams["convcoder"]["f_base"]
        self.f_scale = hparams["convcoder"]["f_scale"]
        self.gf_in = hparams["gf_in"]
        self.gf_n = hparams["gf_n"]
        self.n_id = hparams["convcoder"]["n_id"]
        self.conv_1 = nn.Conv2d(3, self.f_base, 3, padding=1)
        gcv1 = RConv(self.f_base, 3)
        self.rb_1 = nn.ModuleList([copy.deepcopy(gcv1) for _ in range(self.n_layer)])
        self.ln_1 = nn.BatchNorm2d(self.f_base)
        self.pool = nn.MaxPool2d(3, stride=3)
        self.conv_2 = nn.Conv2d(self.f_base, self.f_base * self.f_scale, 1)
        gcv2 = RConv(self.f_base * self.f_scale, 1)
        self.rb_2 = nn.ModuleList([copy.deepcopy(gcv2) for _ in range(self.n_id)])
        self.ln_2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv_1(x)
        for rconv in self.rb_1:
            x = rconv(x) + x
        x = self.ln_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        for rconv in self.rb_2:
            x = rconv(x) + x
        return self.ln_2(x.view(x.size(0), -1))


"""
input is of form (batch, seq, dmodel) where dmodel = 3x3x64
how 2 logits:
 nvm I get it now
transform (batch, -1, dmodel) into (batch, 3, 3, 64)
"""


class PolicyHead(nn.Module):
    def __init__(self, hparams):
        super(PolicyHead, self).__init__()
        self.n_id = hparams["policy"]["n_id"]
        self.n_layer = hparams["policy"]["n_layer"]
        self.f_base = hparams["convcoder"]["f_base"]
        self.f_scale = hparams["convcoder"]["f_scale"]
        self.gf_n = hparams["gf_n"]
        self.f_final = hparams["policy"]["f_final"]
        dcv1 = RConv(self.f_base * self.f_scale, 1)
        self.rb_1 = nn.ModuleList([copy.deepcopy(dcv1) for _ in range(self.n_id)])
        self.ln_1 = nn.BatchNorm2d(self.f_base * self.f_scale)
        self.pool = nn.MaxUnpool2d(3, stride=3)
        self.cv_1 = nn.Conv2d(self.f_base * self.f_scale, self.f_scale, 3, padding=1)
        dcv2 = RConv(self.f_base, 3)
        self.rb_2 = nn.ModuleList([copy.deepcopy(dcv2) for _ in range(self.n_layer)])
        self.ln_2 = nn.BatchNorm2d(self.f_base)
        self.cv_2 = nn.Conv2d(self.f_base, self.f_final, 3, padding=1)
        self.ln_f = nn.BatchNorm2d(self.f_final)
        self.rl_f = nn.ReLU()
        nf = self.gf_n * self.gf_n
        self.fc_o = nn.Linear(self.f_final * nf, nf + 1)

    def forward(self, x):
        x = x[:, -1, :]
        x = x.view(x.size(0), self.f_base * self.f_scale, self.gf_n // 3, self.gf_n // 3)
        for rconv in self.rb_1:
            x = rconv(x) + x
        x = self.cv_1(self.pool(self.ln_1(x)))
        for rconv in self.rb_2:
            x = rconv(x) + x
        x = self.rl_f(self.ln_f(self.cv_2(self.ln_2(x))))
        logits = self.fc_o(x.view(x.size(0), -1))
        return logits


class PolicyHeadB(nn.Module):
    def __init__(self, hparams):
        super(PolicyHeadB, self).__init__()
        self.f_base = hparams["convcoder"]["f_base"]
        self.f_scale = hparams["convcoder"]["f_scale"]
        self.gf_n = hparams["gf_n"]
        self.nx = hparams["trans"]["n_embd"]
        self.ln_f = LayerNorm(self.nx)
        self.rl_f = nn.ReLU()
        nf = self.gf_n * self.gf_n
        self.fc_o = Conv1D(self.f_base * self.f_scale * nf // 9, nf + 1)

    def forward(self, x):
        # (N, seq, dmodel) x (N, seq, nf+1)
        x = self.rl_f(self.ln_f(x))
        logits = self.fc_o(x)
        return logits


class ValueHead(nn.Module):
    def __init__(self, hparams):
        super(ValueHead, self).__init__()
        self.n_id = hparams["value"]["n_id"]
        self.n_layer = hparams["value"]["n_layer"]
        self.f_base = hparams["convcoder"]["f_base"]
        self.f_scale = hparams["convcoder"]["f_scale"]
        self.gf_n = hparams["gf_n"]
        self.f_final = hparams["value"]["f_final"]
        dcv1 = RConv(self.f_base * self.f_scale, 1)
        self.rb_1 = nn.ModuleList([copy.deepcopy(dcv1) for _ in range(self.n_id)])
        self.ln_1 = nn.BatchNorm2d(self.f_base*self.f_scale)
        self.pool = nn.MaxUnpool2d(3)
        self.cv_1 = nn.Conv2d(self.f_base * self.f_scale, self.f_scale, 3, padding=1)
        dcv2 = RConv(self.f_base, 3)
        self.rb_2 = nn.ModuleList([copy.deepcopy(dcv2) for _ in range(self.n_layer)])
        self.ln_2 = nn.BatchNorm2d(self.f_base)
        self.cv_2 = nn.Conv2d(self.f_base, self.f_final, 3, padding=1)
        self.ln_f = nn.BatchNorm2d(self.f_final)
        self.rl_f = nn.ReLU()
        nf = self.gf_n * self.gf_n
        self.fc_o = nn.Linear(self.f_final * nf, 1)
        self.to_o = nn.Tanh()

    def forward(self, x, y=None):
        x = x[:, -1, :]
        x = x.view(x.size(0), self.f_base * self.f_scale, self.gf_n // 3, self.gf_n // 3)
        if y is not None:
            x = x + y
        for rconv in self.rb_1:
            x = rconv(x) + x
        x = self.cv_1(self.pool(self.ln_1(x)))
        for rconv in self.rb_2:
            x = rconv(x) + x
        x = self.rl_f(self.ln_f(self.cv_2(self.ln_2(x))))
        logits = self.fc_o(x.view(x.size(0), -1))
        return self.to_o(logits)


class UniversalDecoderLayer(nn.Module):
    def __init__(self, hparams):
        super(UniversalDecoderLayer, self).__init__()
        self.n_embd = hparams["trans"]["embd"]
        self.n_ctx = hparams["trans"]["ctx"]
        self.n_head = hparams["trans"]["n_head"]
        self.threshold = 0.8
        self.act_max_steps = hparams["trans"]["act_max_steps"]
        self.ln_1 = LayerNorm(self.n_embd)
        self.attn = Attention(self.n_embd, self.n_ctx, self.n_head)
        self.ln_2 = LayerNorm(self.n_embd)
        self.att2 = Attention_(self.n_embd, self.n_ctx, self.n_head)
        self.ln_3 = LayerNorm(self.n_embd)
        self.mlp = MLP(4 * self.n_embd, self.n_embd)
        self.prob = Conv1D(1, self.n_embd)
        self.pact = nn.Sigmoid()
        self.threshold = self.threshold
        self.max_steps = self.act_max_steps

    def ut_function(self, state, step, halting_probability, remainders, n_updates, previous_state, enc_state):
        p = self.pact(self.prob(state))
        still_running = torch.lt(halting_probability, 1.0).float()
        new_halted = torch.gt(halting_probability + p * still_running, self.threshold).float() * still_running
        still_running = torch.le(halting_probability + p * still_running, self.threshold).float() * still_running
        halting_probability += p * still_running
        remainders += new_halted * (1 - halting_probability)
        halting_probability += new_halted * remainders
        n_updates = still_running + new_halted
        update_weights = (p * still_running + new_halted * remainders).unsqueeze(-1)

        x, a1 = self.attn(self.ln_1(state))
        x = x + state
        xo, a2 = self.att2(self.ln_2(x), self.ln_2(enc_state))
        x = x + xo
        m = self.mlp(self.ln_3(x))
        x = x + m
        new_state = (x * update_weights) + previous_state * (1 - update_weights)
        step += 1
        return (x, step, halting_probability, remainders, n_updates, new_state)

    def should_continue(self, u0, u1, halting_probability, u2, n_updates, u3):
        return (torch.lt(halting_probability, self.threshold) & torch.lt(n_updates, self.max_steps)).any()

    def forward(self, x, enc):
        xsiz = x.size()[:-1]
        halting_probability = torch.zeros(xsiz, dtype=torch.float).cuda()
        remainders = torch.zeros(xsiz, dtype=torch.float).cuda()
        n_updates = torch.zeros(xsiz, dtype=torch.float).cuda()
        previous_state = torch.zeros_like(x, dtype=torch.float).cuda()
        step = 0
        while self.should_continue(x, step, halting_probability, remainders, n_updates, previous_state):
            x, step, halting_probability, remainders, n_updates, previous_state = self.ut_function(x, step,
                                                                                                   halting_probability,
                                                                                                   remainders,
                                                                                                   n_updates,
                                                                                                   previous_state, enc)
        return x, (n_updates, remainders)


class Model(nn.Module):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.conv = ConvCoder(hparams)
        self.ut = UniversalDecoderLayer(hparams)
        self.policy = PolicyHead(hparams)
        self.value = ValueHead(hparams)
        self.gframes = []
        self.actions = []
        self.values = [0] * 1
        self.frames = []

    def move_like(self, x, val):
        c = self.conv(x)
        self.frames.append(c)
        uti = torch.stack(self.frames[max(-len(self.frames), -WDW):])
        uto = self.ut(uti, torch.stack(self.gframes))
        pol = self.policy(uto)
        self.actions.append(pol)
        self.values.append(val)
        dvals = [self.values[i] + self.values[i - 1] for i in range(max(len(self.values) - WDW, 1), len(self.values))]
        sft = np.exp(dvals) / np.sum(np.exp(dvals))
        mx = sft.index(max(sft))
        if len(self.frames) > WDW:
            mx += len(self.frames) - WDW - 1
        if self.frames[mx] not in self.gframes:
            self.gframes.append(self.frames[mx])

    def package_state(self, game):
        gs = game.state(nf=min(len(self.frames), WDW))
        for i in range(len(gs)-1, -1, -1):
            if gs[i] in self.gframes:
                self.gframes.pop()
        gf = self.gframes
        gp = self.actions
        gv = self.values
        return (gs, gf, gp, gv)

    def unpack_and_train(self, state, pi, vals):
        gs, gf, gp, gv = state
        self.gframes = gf
        self.actions = gp
        self.values = gv
        for frame in gs:
            c = self.conv(frame)
            self.frames.append(c)
            dvals = [self.values[i] + self.values[i - 1] for i in
                     range(max(len(self.values) - WDW, 1), len(self.values))]
            sft = np.exp(dvals) / np.sum(np.exp(dvals))
            mx = sft.index(max(sft))
            if len(self.frames) > WDW:
                mx += len(self.frames) - WDW - 1
            if self.frames[mx] not in self.gframes:
                self.gframes.append(self.frames[mx])

        uti = torch.stack(self.frames[max(-len(self.frames), -WDW):])
        uto = self.ut(uti, torch.stack(self.gframes))

        for i in range(len(self.frames)-1, -1, -1):
            pol = self.policy(uto[:, i, :])
            val = self.value(uto[:, i, :], y=self.frames[i])
            plfcn = F.cross_entropy()
            vlfcn = F.mse_loss()
            ploss = plfcn(pol, pi[i])
            vloss = vlfcn(val, torch.tensor(vals, dtype=torch.float).unsqueeze(0))
            ploss.backward()
            vloss.backward()
            vals *= -1.0


    def clear(self):
        self.gframes = []
        self.actions = []
        self.values = [0] * 1
        self.frames = []

    def undo(self):
        f = self.frames.pop()
        if f in self.gframes:
            self.gframes.pop()
        self.actions.pop()
        self.values.pop()

    def forward(self, x, residual_value=True, use_globals=False):
        c = self.conv(x)
        self.frames.append(c)
        uti = torch.stack(self.frames[max(-len(self.frames), -WDW):])
        uto = self.ut(uti, torch.stack(self.gframes))
        pol = self.policy(uto)
        val = self.value(uto, y=(c if residual_value else None))
        val = val.view(-1)[0].item()
        self.values.append(val)
        self.actions.append(pol)
        dvals = [self.values[i] + self.values[i - 1] for i in range(max(len(self.values) - WDW, 1), len(self.values))]
        sft = np.exp(dvals) / np.sum(np.exp(dvals))
        mx = sft.index(max(sft))
        if len(self.frames) > WDW:
            mx += len(self.frames) - WDW - 1
        if self.frames[mx] not in self.gframes:
            self.gframes.append(self.frames[mx])
        return pol, val


class Agent(MCTS):
    def __init__(self, hparams):
        super(Agent, self).__init__(game=Game())
        self.game = Game()
        self.model = Model(hparams)

    def heuristic_value(self, game, s, a):
        N, a_ = self.nodes.get(s, (1, None))
        p = (1e-5, 0, 0)
        Ni, W, P = p if a_ is None else a_.get(a, p)
        return W * (1.0 / Ni) + self.C * P * np.sqrt(N) / (1 + Ni)

    def record(self, s, a, r):
        n = self.nodes.get(s, None)
        if n is None:
            a_ = {}
            a_[a] = (1, r, 1.0)
            self.nodes[s] = (1, a_)
        else:
            N, a_ = self.nodes[s]
            N += 1
            n_a = a_.get(a, (0, 0, 0))
            n_a[0] += 1
            n_a[1] += r
            n_a[2] = n_a[0] / N
            a_[a] = n_a
            self.nodes[s] = (N, a_)

    def playout_value(self, game, alt=False):
        s = game.state(nf=WDW, to_hash=True, flip_states=alt)
        if game.over():
            self.record(s, NN, -game.score())
            return -game.score()
        elif self.nodes.get(s, None) is None:
            P, v = self.model(game.state(flip_states=alt)[-1])
            P = P.cpu()
            v = v.cpu()
            g = {}
            for i in game.valid_moves():
                g[i] = (0, v, P.view(-1).numpy()[i])
            self.nodes[s] = (1, g)
            return -v
        ahd = {}

        for a in game.valid_moves():
            ahd[a] = -self.heuristic_value(game, s, a)
        move = max(ahd, key=ahd.get)
        self.model.move_like(game.state(flip_states=alt), ahd[move])
        game.make_move(move)
        value = -self.playout_value(game)
        game.undo_move()
        self.model.undo()
        self.record(s, move, value)
        return value

    def monte_carlo_value(self, game, N=100):
        scores = [self.playout_value(game) for i in range(0, N)]
        return np.mean(scores)

    def pi(self, game):
        s = game.state(nf=WDW, to_hash=True)
        N, a_ = self.nodes[s]
        pol = np.zeros((1, NN + 1), dtype=np.float)
        for i in range(len(pol)):
            Ni, W, P = a_.get(i, (-1, 0, 0))
            if Ni != -1:
                pol[0, i] = Ni / N
        return pol

    def train_net(self, examples):
        nnet = copy.deepcopy(self.model)
        optm = optim.Adam(nnet.parameters(), lr=1e-4)
        optm.zero_grad()
        for examp in examples:
            nnet.unpack_state_and_train(examp[0], examp[1], examp[2])
            optm.step()
        return nnet

    def train_iter(self, game, give_examp=False):
        examples = []
        for i in range(100):
            examples += self.exec_exec(game)
            self.model.clear()
        new_net = self.train_net(examples)
        rival = Agent(game)
        rival.model = new_net
        if Agent.fight(rival, self, give_examp=give_examp):
            self.model = new_net

    @staticmethod
    def fight(a1, a2, give_examp=False):
        sum = 0
        for i in range(100):
            sum += Agent.play_game(a1, a2, i % 2 == 0, give_examp)
        return sum >= 5

    @staticmethod
    def play_game(a1, a2, a_first, print_out=False):
        game = Game()
        while True:
            if print_out:
                game.print_game()
            if a_first:
                for _ in range(100):
                    a1.playout_value(game)
                m = [mcts for mcts in a1.pi(game).squeeze(0)]
                mv = m.index(max(m))
                game.make_move(mv)
                a1.model.forward(game.state(nf=1))
                a2.model.forward(game.state(nf=1, flip_states=True))
                a_first = not a_first
            elif not a_first:
                for _ in range(100):
                    a2.playout_value(game, flip_states=True)
                m = [mcts for mcts in a1.pi(game).squeeze(0)]
                mv = m.index(max(m))
                game.make_move(mv)
                a1.model.forward(game.state(nf=1))
                a2.model.forward(game.state(nf=1, flip_states=True))
                a_first = not a_first
            if game.over():
                return game.score()

    def exec_exec(self, game):
        examples = []
        game = Game()
        p = []
        while True:
            for _ in range(100):
                self.playout_value(game)
            mctspi = self.pi(game)
            p.append(torch.tensor(mctspi, dtype=torch.float, device=torch.device("cuda")))
            examples.append([self.model.package_states(), p[0:min(len(p), WDW)], None])
            mctspi = [mcts for mcts in mctspi.squeeze(0)]
            a = random.choice(len(mctspi), p=mctspi)
            game.make_move(a)
            self.model.forward(game.state(nf=1))
            if game.over():
                examples = self.assign_reward(examples, game.score())
                self.nodes = {}
                a_ = {}
                self.nodes[game.empty] = (1, a_)

    def assign_reward(self, examples, score):
        for i in range(len(examples) - 1, 0, -1):
            examples[2] = score
            score *= -1.0
        return examples


import torch.cuda as tcuda


def main():
    h = Game()
    h.print_game()
    device = torch.device("cuda" if tcuda.is_available() else "cpu")
    print("babu")
    main_agent = Agent(get_default_hparams())
    main_agent.model.to(device)
    dummy_agent = Agent(get_default_hparams())
    dummy_agent.model.to(device)
    Agent.play_game(main_agent, dummy_agent, True, True)
    default_game = Game(device)

    for i in range(10000):
        main_agent.train_iter(default_game, True)


if __name__ == '__main__':
    main()

"""
a = max
{gamestate: a{W} / gamestate: a{N} + C * sqrt(log(Ntot) / gamestate: a{N})}
if gamestate is game.over():
    P = 0
    v = game.outcome()
elif hash(gamestate) is not in gamestates:
    P, v = nnet(O, gamestate)
    P = mask(P, validmoves)
else:
    gamestate = game.makemove(a)
    repeat
for previous gamestate: a:
N = N + 1
W = W + v
for new gamestate: a
if P[a] > 0:
    N = 1
    W = v
    append()
"""