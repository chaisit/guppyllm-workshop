"""
benchmark.py — วัดความเร็ว inference บนเครื่องนี้

Usage:
    python benchmark.py
    python benchmark.py --quantize     # เทียบ FP32 vs INT8
"""
import argparse
import time

import torch

from guppylm.inference import GuppyInference

PROMPTS = [
    "hi guppy",
    "are you hungry?",
    "what is water",
    "tell me a joke",
    "do you like light",
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def run_benchmark(engine, label: str):
    print(f"\n=== {label} ===")
    print(f"{'prompt':22} | {'time':>7} | {'tokens':>7} | {'tok/s':>7}")
    print("-" * 55)

    total_time, total_tokens = 0.0, 0
    for p in PROMPTS:
        t0 = time.time()
        r = engine.chat_completion([{"role": "user", "content": p}], max_tokens=64)
        dt = time.time() - t0
        text = r["choices"][0]["message"]["content"]
        n_tok = len(engine.tokenizer.encode(text).ids)

        total_time += dt
        total_tokens += n_tok
        rate = n_tok / dt if dt > 0 else 0
        print(f"{p:22} | {dt:6.2f}s | {n_tok:7d} | {rate:7.1f}")

    print("-" * 55)
    avg = total_tokens / total_time if total_time > 0 else 0
    print(f"เฉลี่ย: {avg:.1f} tokens/second")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--quantize", action="store_true",
                    help="เทียบ FP32 กับ INT8 dynamic quantization")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    engine = GuppyInference(args.checkpoint, args.tokenizer, device)
    fp32_rate = run_benchmark(engine, "FP32 (baseline)")

    if args.quantize:
        if device != "cpu":
            print("\n[!] Dynamic quantization ออกแบบมาสำหรับ CPU — ผลบน GPU อาจไม่ตรงคาด")
        engine.model = torch.quantization.quantize_dynamic(
            engine.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        int8_rate = run_benchmark(engine, "INT8 (dynamic quantization)")

        if fp32_rate > 0:
            print(f"\nSpeedup: {int8_rate / fp32_rate:.2f}x")
        print("\n[!] อย่าลืมตรวจคุณภาพคำตอบด้วย ไม่ใช่แค่ความเร็ว")


if __name__ == "__main__":
    main()
