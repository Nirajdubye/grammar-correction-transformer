import os
import math
import torch
import torch.nn as nn
from collections import Counter
from tokenizers import ByteLevelBPETokenizer

MAX_LEN = 500


# ---------------------------------------------------------------------------
# 1. Architecture Components
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'.
    Generates fixed sine and cosine frequencies across embedding dimensions.
    """
    def __init__(self, d_model: int, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """
    Seq2Seq Transformer Encoder-Decoder with Sinusoidal Positional Encoding
    specifically tailored for character/subword token level grammar error correction.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))

        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(src.device)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc(out)


# ---------------------------------------------------------------------------
# 2. Model & Tokenizer Loader
# ---------------------------------------------------------------------------
class GrammarCorrector:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/epoch_20.pt",
        tokenizer_dir: str = "tokenizer",
        device: str = None
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        vocab_file = os.path.join(tokenizer_dir, "vocab.json")
        merges_file = os.path.join(tokenizer_dir, "merges.txt")

        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            raise FileNotFoundError(f"Tokenizer files not found in: {tokenizer_dir}")

        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        self.pad_idx = self.tokenizer.token_to_id("<pad>")
        self.sos_idx = self.tokenizer.token_to_id("<sos>")
        self.eos_idx = self.tokenizer.token_to_id("<eos>")
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.model = TransformerModel(self.vocab_size).to(self.device)

        self.checkpoint_loaded = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if isinstance(state_dict, dict) and self.device.type == "cpu":
                state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.checkpoint_loaded = True
            print(f"Loaded checkpoint from: {checkpoint_path} on {self.device}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Operating in demo/mock mode.")

    def encode(self, text: str) -> list:
        return [self.sos_idx] + self.tokenizer.encode(text).ids[:MAX_LEN - 2] + [self.eos_idx]

    def decode(self, ids: list) -> str:
        ids_filtered = [i for i in ids if i not in (self.pad_idx, self.sos_idx, self.eos_idx)]
        return self.tokenizer.decode(ids_filtered)

    @staticmethod
    def has_repeat_ngram(sequence: list, token: int, n: int = 3) -> bool:
        if len(sequence) < n - 1:
            return False
        target = tuple(sequence[-(n - 1):] + [token])
        for i in range(len(sequence) - n + 1):
            if tuple(sequence[i:i + n]) == target:
                return True
        return False

    @staticmethod
    def repetition_score(tokens: list) -> float:
        if len(tokens) < 4:
            return 0.0
        counts = Counter(tokens)
        return counts.most_common(1)[0][1] / len(tokens)

    def generate(self, input_text: str, max_additional_tokens: int = 50) -> str:
        """
        Generates corrected text using autoregressive greedy decoding with:
        - 3-gram repetition blocking
        - Repetition score safety checks (fallback to original text)
        - Short/empty sentence fallback
        """
        if not input_text or not input_text.strip():
            return input_text

        if not self.checkpoint_loaded:
            # Fallback for when weights are not yet downloaded
            return input_text

        src_ids = self.encode(input_text.strip())
        src = torch.tensor([src_ids], device=self.device)
        tgt = torch.tensor([[self.sos_idx]], device=self.device)

        generated = []
        max_steps = min(len(src_ids) + max_additional_tokens, MAX_LEN)

        with torch.no_grad():
            for _ in range(max_steps):
                output = self.model(src, tgt)
                logits = output[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).item()

                if next_token == self.eos_idx:
                    break

                # 3-gram repetition blocking
                if self.has_repeat_ngram(generated, next_token, n=3):
                    top_tokens = torch.topk(logits, 5).indices[0].tolist()
                    found = False
                    for t in top_tokens:
                        if t == self.eos_idx:
                            continue
                        if not self.has_repeat_ngram(generated, t, n=3):
                            next_token = t
                            found = True
                            break
                    if not found:
                        break

                generated.append(next_token)
                tgt = torch.cat([tgt, torch.tensor([[next_token]], device=self.device)], dim=1)

        result = self.decode(generated).strip()

        # Repetition sanity check
        if self.repetition_score(generated) > 0.85:
            return input_text

        # Length sanity check: prevent empty or collapsed output
        if not result or len(result.split()) < max(1, len(input_text.split()) - 3):
            return input_text

        return result


# ---------------------------------------------------------------------------
# 3. Interactive CLI Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Grammar Correction Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_20.pt", help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Path to tokenizer folder")
    args = parser.parse_args()

    corrector = GrammarCorrector(checkpoint_path=args.checkpoint, tokenizer_dir=args.tokenizer)
    print("\n🧠 Grammar Correction CLI Ready (Type 'exit' to quit)\n" + "=" * 55)

    while True:
        try:
            user_input = input("Input: ").strip()
            if not user_input or user_input.lower() == "exit":
                break
            output = corrector.generate(user_input)
            print("Output:", output)
            print("-" * 55)
        except (KeyboardInterrupt, EOFError):
            break
