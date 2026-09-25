

Yes. If you mean **relative to your custom model = 100%**, your results look like this:

| Metric              |  Custom | PyTorch | PyTorch relative |
| ------------------- | ------: | ------: | ---------------: |
| **Train BLEU**      |   94.66 |   92.71 |       **97.94%** |
| **Validation BLEU** |   23.82 |   19.37 |       **81.32%** |
| **Test BLEU**       |   24.26 |   20.25 |       **83.47%** |
| **Peak GPU memory** | 1675 MB | 1884 MB |      **112.48%** |
| **Avg epoch**       |    90 s |    96 s |      **106.67%** |

Or, looking at **your model relative to PyTorch**:

* **Train BLEU:** 2.10% higher
* **Validation BLEU:** 22.87% higher
* **Test BLEU:** 19.80% higher
* **GPU memory:** **11.09% lower**
* **Epoch time:** **6.25% faster**

And yes — **this is strong evidence that your implementation is actually working.** The particularly useful signal is that the advantage isn't only on training BLEU; it carries through to **validation and test**.

I would now treat the custom Transformer as a **valid working 2017 Transformer implementation**, and move on to understanding *why* the runtime characteristics differ rather than questioning whether the implementation works.


BLEU comparison

Your custom Transformer versus the PyTorch nn.Transformer benchmark.

metric	custom	pytorch
Train	94.66	92.71
Validation	23.82	19.37
Test	24.26	20.25