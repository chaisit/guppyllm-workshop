"""
app.py — Gradio Web UI สำหรับ GuppyLM ที่เทรนเอง

Usage:
    pip install gradio
    python app.py
    -> เปิด http://localhost:7860
"""
import torch
import gradio as gr

from guppylm.inference import GuppyInference


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
print(f"Loading model on {DEVICE}...")
engine = GuppyInference(
    checkpoint_path="checkpoints/best_model.pt",
    tokenizer_path="data/tokenizer.json",
    device=DEVICE,
)
print("Model loaded.")


def chat(message, history, temperature, top_k, max_tokens):
    """
    หมายเหตุ: ส่งเฉพาะข้อความล่าสุด ไม่ส่ง history
    เพราะ context ของ GuppyLM จำกัดที่ 128 tokens
    """
    r = engine.chat_completion(
        [{"role": "user", "content": message}],
        temperature=float(temperature),
        top_k=int(top_k),
        max_tokens=int(max_tokens),
    )
    return r["choices"][0]["message"]["content"]


demo = gr.ChatInterface(
    fn=chat,
    title="My GuppyLM",
    description=(
        f"LLM 8.7M parameters ที่เทรนเอง · รันบน {DEVICE} · "
        "ไม่ต้องใช้ internet ไม่ต้องมี API key"
    ),
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(1, 100, value=50, step=1, label="Top-K"),
        gr.Slider(16, 128, value=64, step=8, label="Max tokens"),
    ],
    examples=[
        "hi guppy",
        "are you hungry?",
        "tell me a joke",
        "what is light",
    ],
)

if __name__ == "__main__":
    demo.launch()
