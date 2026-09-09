# 💻 Local Scripts

Scripts สำหรับรัน GuppyLM บนเครื่องตัวเอง หลังจากดาวน์โหลด checkpoint จาก Colab แล้ว

> 📌 **หมายเหตุ:** repo GuppyLM มี CLI ในตัวอยู่แล้ว (`python -m guppylm chat`) ที่รันได้ทันที
> สคริปต์ในโฟลเดอร์นี้ (`chat.py`, `app.py`, `benchmark.py`) เป็นเวอร์ชันที่**เราเขียนเองในเวิร์กช็อป** โดย wrap รอบ `GuppyInference` เพื่อเรียนรู้การทำ device-agnostic, การสร้าง Web UI และการวัด benchmark — ทั้งสองแบบใช้ engine ตัวเดียวกัน

## การติดตั้ง

```bash
python -m venv guppy-env
source guppy-env/bin/activate        # Windows: guppy-env\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers safetensors gradio

git clone https://github.com/arman-bd/guppylm.git my-guppy
cd my-guppy
```

## โครงสร้างไฟล์ที่ต้องมี

```
my-guppy/
├── guppylm/                  # จาก git clone
├── checkpoints/
│   └── best_model.pt         # 📥 จาก Colab
├── data/
│   └── tokenizer.json        # 📥 จาก Colab
├── chat.py                   # จากโฟลเดอร์นี้
├── app.py                    # จากโฟลเดอร์นี้
└── benchmark.py              # จากโฟลเดอร์นี้
```

## Scripts

| ไฟล์ | ใช้ทำอะไร | คำสั่ง |
|---|---|---|
| `chat.py` | CLI chat แบบ device-agnostic | `python chat.py` |
| `app.py` | Gradio Web UI | `python app.py` → http://localhost:7860 |
| `benchmark.py` | วัดความเร็ว + เทียบ quantization | `python benchmark.py --quantize` |

### chat.py

```bash
python chat.py                                # interactive
python chat.py --prompt "tell me a joke"      # ถามครั้งเดียว
python chat.py --device cpu --temperature 0.9 # กำหนดค่าเอง
```

### app.py

```bash
pip install gradio
python app.py
```

### benchmark.py

```bash
python benchmark.py              # วัด FP32 อย่างเดียว
python benchmark.py --quantize   # เทียบ FP32 vs INT8
```

## ⚠️ จุดที่มักพลาด

| Error | วิธีแก้ |
|---|---|
| `UnpicklingError: Weights only load failed` | PyTorch 2.6+ — ต้องใช้ `weights_only=False` |
| `Attempting to deserialize object on a CUDA device` | ต้องใช้ `map_location='cpu'` |
| `ModuleNotFoundError: No module named 'guppylm'` | ต้องรันจาก directory ที่มีโฟลเดอร์ `guppylm/` |
| `FileNotFoundError: data/tokenizer.json` | ตรวจว่าวางไฟล์จาก Colab ถูกตำแหน่ง |

ดูรายละเอียดเพิ่มที่ [Troubleshooting Guide](../../docs/08-troubleshooting.md)
