# File formats

## The network for other runtimes (`TAFLNET1`)

`python -m tafl.export --model <checkpoint> --out net.bin` writes the policy-value
network in a small binary file that a program in another language can read without
PyTorch: the article's browser port reads it in JavaScript. The file holds the network
as the engine evaluates it on a GPU, `Net.fused()`: every batch norm folded into the
layer before it, and every weight rounded to IEEE float16. It keeps only what a player
needs: the stem, the residual blocks, the move policy (the first 40 output planes of
`policy.out`; the 40 planes of the opponent-reply policy, a training target, are left
out) and the whole value head.

All numbers are little-endian. A 64-byte header comes first:

| Offset | Type | Field | Value for step 468,000 |
| --- | --- | --- | --- |
| 0 | 8 bytes | magic | `TAFLNET1` |
| 8 | u32 | input planes | 20 |
| 12 | u32 | board side | 11 |
| 16 | u32 | residual blocks B | 10 |
| 20 | u32 | channels C | 128 |
| 24 | u32 | squeeze-excitation ratio | 4 |
| 28 | u32 | squeeze-excitation hidden units S = max(C / ratio, 8) | 32 |
| 32 | u32 | head channels H | 32 |
| 36 | u32 | value hidden units V | 128 |
| 40 | u32 | policy planes | 40 |
| 44 | u32 | value outputs | 3 |
| 48 | u32 | ply plane | 19 |
| 52 | f32 | ply scale | 1/128 |
| 56 | f32 | moves-left scale | 100 |
| 60 | u32 | N, the number of weights that follow | 3,166,059 |

Then come N float16 weights and nothing else, so the file is 64 + 2N bytes: 6,332,182
for the current network. They are the tensors below, in this order, each in PyTorch's
row-major order: a convolution's weights are `[out][in][ky][kx]` (for a 1x1
convolution simply `[out][in]`) and a linear layer's `[out][in]`. The names are those
of a fused `Net`'s state dict.

| Tensor | Shape | Weights |
| --- | --- | --- |
| `stem.weight`, `stem.bias` | [C, 20, 3, 3], [C] | 23,168 |
| for each block b: `blocks.b.conv1.weight`, `.conv1.bias` | [C, C, 3, 3], [C] | 147,584 |
| `blocks.b.conv2.weight`, `.conv2.bias` | [C, C, 3, 3], [C] | 147,584 |
| `blocks.b.se1.weight`, `.se1.bias` | [S, 2C], [S] | 8,224 |
| `blocks.b.se2.weight`, `.se2.bias` | [2C, S], [2C] | 8,448 |
| `policy.conv_g.weight`, `.bias` | [H, C, 1, 1], [H] | 4,128 |
| `policy.fc_g.weight`, `.bias` | [H, 2H], [H] | 2,080 |
| `policy.conv_p.weight`, `.bias` | [H, C, 1, 1], [H] | 4,128 |
| `policy.out.weight`, `.bias` (the move planes only) | [40, H, 1, 1], [40] | 1,320 |
| `value_conv.weight`, `.bias` | [H, C, 1, 1], [H] | 4,128 |
| `value_fc.weight`, `.bias` | [V, 2H], [V] | 8,320 |
| `value_out.weight`, `.bias` | [3, V], [3] | 387 |

The forward pass, for one position:

1. The input is the engine's 2,420 bytes (`crates/tafl-engine/src/encode.rs`): 20
   planes of 121 squares, square = row * 11 + column from a1. Every byte becomes a
   number; the ply plane (19), which holds the ply count up to 255, is multiplied by
   the ply scale.
2. Every 3x3 convolution is PyTorch's: output channel o at square (y, x) is its bias
   plus the sum over input channels i and offsets (ky, kx) of
   `W[o][i][ky][kx] * in[i][y + ky - 1][x + kx - 1]`, with zeros off the board.
3. Stem: `x = relu(conv(x))`.
4. Each block: `y = relu(conv1(x))`, `y = conv2(y)`. Pool y over the board into 2C
   numbers, the mean of each channel and then the maximum of each. `t = se2(relu(se1(pooled)))`
   gives a gate (the first C numbers) and a shift (the last C). Then
   `x = relu(x + y * sigmoid(gate) + shift)`, channel by channel.
5. Policy: `g = relu(conv_g(x))`, pooled the same way (mean then max, 2H numbers), and
   `p = relu(conv_p(x) + fc_g(pooled g))`, the H numbers of `fc_g` added to every
   square of their channel. Then `out(p)` gives 40 planes of 121 logits: the logit of
   action `plane * 121 + from square`, where `plane = direction * 10 + distance - 1`
   and the directions are north, south, east and west.
6. Value: `v = relu(value_conv(x))`, pooled (2H numbers), `o = value_out(relu(value_fc(pooled v)))`.
   For the side to move, `o[0]` is the logit of a win, `tanh(o[0] / 2)` the expected
   result in [-1, 1] that the search uses, `tanh(o[1])` the searched value the network
   learned to predict (`q`), and `softplus(o[2])` times the moves-left scale the
   number of plies left.

`python/tests/test_export.py` reads a file with numpy alone and computes this forward
pass, which must agree with PyTorch.

## Reference outputs

`python -m tafl.export ... --reference ref.json [--games 4 --positions 64 --seed 1]`
also plays a few games between two copies of the network (each move the policy's best
legal move, or in the first 4 plies and a fifth of the others a sample from it), keeps
the start position and a random sample of the others, and writes JSON for checking
another runtime. `net_sha256` identifies the network file. Each entry of `positions`
has the game, ply and moves that led to it, `planes` (the 2,420 input bytes in base64,
written by the engine's encoder through the Python bindings), `actions` (the legal
moves' action indices) and the outputs of three versions of the network, all run by
PyTorch on the CPU:

- `stored`: float32 on the file's weights (the float16 values, widened);
- `checkpoint`: float32 on the checkpoint, batch norms not folded;
- `fp16`: the fused float16 network, as the engine runs it on a GPU.

Each holds `logits` (one per legal action, in the order of `actions`), `argmax` (the
action with the largest of them), `value_logit`, `q`, `moves_left` and `value`
(`tanh(value_logit / 2)`); `stored` also has all 4,840 logits of the first two
positions as `all_logits`.
