"""
chat.py — Local GuppyLM chat interface
รองรับ cuda / mps / cpu อัตโนมัติ

Usage:
    python chat.py
    python chat.py --prompt "tell me a joke"
"""
import argparse
import time

import torch

from guppylm.inference import GuppyInference


def pick_device() -> str:
    """เลือก device ที่ดีที่สุดที่มีในเครื่องนี้: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def build_engine(checkpoint: str, tokenizer: str, device: str) -> GuppyInference:
    print(f"Device: {device}")
    print("Loading model...")
    t0 = time.time()
    engine = GuppyInference(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        device=device,
    )
    print(f"Model loaded in {time.time() - t0:.2f}s\n")
    return engine


def ask(engine, message: str, temperature: float, top_k: int, max_tokens: int):
    t0 = time.time()
    r = engine.chat_completion(
        [{"role": "user", "content": message}],
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    answer = r["choices"][0]["message"]["content"]
    return answer, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Chat with your locally-trained GuppyLM")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--prompt", default=None, help="ถามครั้งเดียวแล้วจบ")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    device = args.device or pick_device()
    engine = build_engine(args.checkpoint, args.tokenizer, device)

    # โหมดถามครั้งเดียว
    if args.prompt:
        answer, dt = ask(engine, args.prompt, args.temperature,
                         args.top_k, args.max_tokens)
        print(f"You>   {args.prompt}")
        print(f"Guppy> {answer}")
        print(f"       ({dt:.2f}s)")
        return

    # โหมด interactive
    print("Chat with Guppy (type 'quit' to exit)")
    print("-" * 45)
    while True:
        try:
            msg = input("You>   ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if msg.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not msg:
            continue

        answer, dt = ask(engine, msg, args.temperature,
                         args.top_k, args.max_tokens)
        print(f"Guppy> {answer}")
        print(f"       ({dt:.2f}s)\n")


if __name__ == "__main__":
    main()
