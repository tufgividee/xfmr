import torch
from torch import nn


class Translator:
    def __init__(
        self,
        model: nn.Module,
        src_tokenizer,
        tgt_tokenizer,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.tgt_sos_idx = tgt_tokenizer.token_to_id("[SOS]")
        self.tgt_eos_idx = tgt_tokenizer.token_to_id("[EOS]")

    @torch.no_grad()
    def translate(
        self,
        text: str,
        max_length: int = 50,
    ) -> str:
        self.model.eval()

        src_ids = self.src_tokenizer.encode(text).ids

        src_tokens = torch.tensor(
            [src_ids],
            dtype=torch.long,
            device=self.device,
        )

        tgt_tokens = torch.tensor(
            [[self.tgt_sos_idx]],
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_length):
            output = self.model(src_tokens, tgt_tokens)

            next_token_id = output[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            tgt_tokens = torch.cat(
                [tgt_tokens, next_token_id],
                dim=1,
            )

            if next_token_id.item() == self.tgt_eos_idx:
                break

        generated_ids = tgt_tokens.squeeze(0).tolist()

        return self.tgt_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )