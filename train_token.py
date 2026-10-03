import os
import argparse
from tokenizers import ByteLevelBPETokenizer


def train_tokenizer(corpus_path: str = "corpus.txt", output_dir: str = "tokenizer", vocab_size: int = 32000, min_frequency: int = 2):
    """
    Trains a Byte-Level Byte-Pair Encoding (BPE) tokenizer on the specified corpus.
    Outputs: vocab.json and merges.txt in output_dir.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 Training Byte-Level BPE tokenizer on {corpus_path}...")

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<sos>",
            "<eos>",
            "<mask >"
        ]
    )

    print(f"💾 Saving tokenizer to {output_dir}...")
    tokenizer.save_model(output_dir)
    print(f"✅ Tokenizer successfully saved at: {output_dir}")
    return tokenizer


def test_tokenizer(tokenizer_dir: str = "tokenizer", test_sentence: str = "he dont know this sentence"):
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)

    encoded = tokenizer.encode(test_sentence)
    print("\n🧪 Test Tokenizer Result:")
    print("Input Text:", test_sentence)
    print("Tokens:    ", encoded.tokens)
    print("IDs:       ", encoded.ids)
    print("Decoded:   ", tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE Tokenizer for GEC")
    parser.add_argument("--corpus", type=str, default="corpus.txt", help="Path to text corpus file")
    parser.add_argument("--output_dir", type=str, default="tokenizer", help="Output directory for vocab and merges")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--test", action="store_true", help="Run test on sample sentence")
    args = parser.parse_args()

    if args.test or not os.path.exists(args.corpus):
        print(f"Running tokenizer test with existing files in {args.output_dir}...")
        test_tokenizer(args.output_dir)
    else:
        train_tokenizer(args.corpus, args.output_dir, args.vocab_size)
        test_tokenizer(args.output_dir)
