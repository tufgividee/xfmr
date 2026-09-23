| Component                         | Your custom Transformer | PyTorch `nn.Transformer`     |
| --------------------------------- | ----------------------- | ---------------------------- |
| Encoder/decoder layers            | 6 / 6                   | 6 / 6                        |
| `d_model`                         | 512                     | 512                          |
| Heads                             | 8                       | 8                            |
| FFN                               | 512 → 2048 → 512        | Same                         |
| FFN activation                    | ReLU                    | ReLU                         |
| Attention scaling                 | `1/√d_k`                | `1/√d_k`                     |
| Post-LN                           | **Yes**                 | **Yes** (`norm_first=False`) |
| Embedding √d scaling              | **Yes**                 | **Yes**                      |
| Positional encoding               | Same implementation     | Same implementation          |
| Input embedding dropout           | 0.1                     | 0.1                          |
| Sublayer/residual dropout         | 0.1                     | 0.1                          |
| **Attention probability dropout** | **No**                  | **Yes, 0.1**                 |
| **Final encoder LayerNorm**       | **No**                  | **Yes**                      |
| **Final decoder LayerNorm**       | **No**                  | **Yes**                      |
| Padding masks                     | Same semantics          | Same semantics               |
| Causal target mask                | Same semantics          | Same semantics               |
| Decoder cross-attention           | Yes                     | Yes                          |
| Output weight tying               | **Yes**                 | **Yes**                      |
| Output projection bias            | Yes                     | Yes                          |
