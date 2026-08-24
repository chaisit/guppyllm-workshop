[⬅️ กลับหน้าหลัก](../../README.md)

---

# 📓 Colab Cells พร้อมใช้

> คัดลอกทีละ cell ไปวางใน Google Colab ตามลำดับ

## 🔢 ลำดับการรัน

```mermaid
flowchart LR
    C1["1️⃣ Setup"] --> C2["2️⃣ Mount Drive"]
    C2 --> C3["3️⃣ Data"]
    C3 --> C4["4️⃣ Train"]
    C4 --> C5["5️⃣ Test"]
    C5 --> C6["6️⃣ Export"]
    C6 --> C7["7️⃣ Download"]

    style C2 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style C6 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

> ⚠️ **Cell 2 (Mount Drive) ห้ามข้าม** — ถ้าข้าม checkpoint จะหายเมื่อ session ตาย

---

## Cell 1 — Setup และตรวจ GPU

```python
# ตรวจว่าได้ GPU หรือไม่
!nvidia-smi

# Clone repository
!git clone https://github.com/arman-bd/guppylm.git
%cd guppylm

# ติดตั้ง dependencies
!pip install -q tokenizers datasets safetensors

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Cell 2 — Mount Google Drive ⚠️ สำคัญที่สุด

```python
from google.colab import drive
drive.mount('/content/drive')

import os

WORKSHOP_DIR = '/content/drive/MyDrive/guppylm_workshop'
CKPT_DIR = f'{WORKSHOP_DIR}/checkpoints'
EXPORT_DIR = f'{WORKSHOP_DIR}/export'

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"Workshop dir: {WORKSHOP_DIR}")
print(f"Checkpoints:  {CKPT_DIR}")
print(f"Export:       {EXPORT_DIR}")
print("\n>>> ตรวจว่า path ขึ้นต้นด้วย /content/drive/ แล้วยกมือ <<<")
```

> 🔴 **ถ้า path ไม่ขึ้นต้นด้วย `/content/drive/` แปลว่า mount ไม่สำเร็จ — อย่าเทรนต่อ**

---

## Cell 3 — เตรียม Data และ Tokenizer

```python
# ใช้ script สำเร็จรูปของ repo
!python -m guppylm.prepare_data

# ตรวจผลลัพธ์
import os
for f in sorted(os.listdir('data')):
    size = os.path.getsize(f'data/{f}')
    print(f"{f:25} {size/1024:>10,.1f} KB")
```

### Cell 3b — สำรวจ dataset (ถ้าต้องการ)

```python
from datasets import load_dataset
from collections import Counter

ds = load_dataset("arman-bd/guppylm-60k-generic")
print(ds)

# ตัวอย่างข้อมูล
for i in range(3):
    ex = ds['train'][i]
    print(f"\n--- {i+1} ---")
    print(f"category: {ex['category']}")
    print(f"input:    {ex['input']}")
    print(f"output:   {ex['output']}")

# วัด diversity
outputs = ds['train']['output']
unique, total = len(set(outputs)), len(outputs)
print(f"\nUniqueness: {unique:,}/{total:,} = {unique/total*100:.1f}%")
print(f"ซ้ำเฉลี่ย: {total/unique:.1f} ครั้ง/คำตอบ")
```

### Cell 3c — ทดสอบ tokenizer

```python
from tokenizers import Tokenizer
tk = Tokenizer.from_file("data/tokenizer.json")

for t in ["hi guppy", "are you hungry?", "quantum entanglement theory"]:
    enc = tk.encode(t)
    print(f"\n{t!r}")
    print(f"  {len(enc.ids)} tokens: {enc.tokens}")
```

---

## Cell 4 — Training ⏰ กด Run ก่อนพักเที่ยง

```python
# ตรวจ config ก่อนเทรน
from guppylm.config import GuppyConfig, TrainConfig
from guppylm.model import GuppyLM

mc = GuppyConfig()
tc = TrainConfig()

print("=== Model Config ===")
for k, v in vars(mc).items():
    print(f"  {k:15} = {v}")

print("\n=== Train Config ===")
for k, v in vars(tc).items():
    print(f"  {k:15} = {v}")

m = GuppyLM(mc)
print(f"\n{m.param_summary()}")
```

```python
# ⚠️ ตั้ง output_dir ให้ชี้ไป Google Drive ก่อนเทรน
# วิธีที่ 1: แก้ค่าใน config.py โดยตรง
# วิธีที่ 2: sed แทนที่ (เร็วกว่าสำหรับห้องเรียน)

!sed -i "s|output_dir: str = \"checkpoints\"|output_dir: str = \"{CKPT_DIR}\"|" guppylm/config.py
!grep output_dir guppylm/config.py
```

```python
# 🚀 เริ่มเทรน — ใช้เวลา ~5 นาทีบน T4 ที่ว่าง
!python -m guppylm.train
```

**สิ่งที่ควรเห็น:**
- loss เริ่มต้นราว **8.3** (= ln(4096))
- loss ลดลงเรื่อย ๆ
- มีข้อความ save checkpoint เป็นระยะ

---

## Cell 5 — ทดสอบโมเดล (ยังอยู่บน Colab)

```python
from guppylm.inference import GuppyInference

engine = GuppyInference(
    checkpoint_path=f'{CKPT_DIR}/best_model.pt',
    tokenizer_path='data/tokenizer.json',
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

for q in ["hi guppy", "are you hungry?", "tell me a joke", "what is water"]:
    r = engine.chat_completion([{"role": "user", "content": q}])
    print(f"You>   {q}")
    print(f"Guppy> {r['choices'][0]['message']['content']}\n")
```

### Cell 5b — ทดลอง sampling parameters

```python
q = "tell me about water"

for temp in [0.2, 0.7, 1.5]:
    r = engine.chat_completion(
        [{"role": "user", "content": q}], temperature=temp
    )
    print(f"temp={temp}: {r['choices'][0]['message']['content']}")

print()
for k in [1, 10, 50]:
    r = engine.chat_completion(
        [{"role": "user", "content": q}], top_k=k
    )
    print(f"top_k={k}: {r['choices'][0]['message']['content']}")
```

---

## Cell 6 — Export Checkpoint ⭐

```python
import torch, json, os, shutil

# โหลด checkpoint
ckpt = torch.load(
    f'{CKPT_DIR}/best_model.pt',
    map_location='cpu',
    weights_only=False,   # ⚠️ จำเป็นสำหรับ PyTorch 2.6+
)
print("Keys ใน checkpoint:", list(ckpt.keys()))

# 1) save state_dict (พกพาได้ดีกว่า save ทั้ง model)
torch.save(ckpt['model_state_dict'], f'{EXPORT_DIR}/pytorch_model.bin')

# 2) save config เป็น JSON
cfg = ckpt['config']
cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
with open(f'{EXPORT_DIR}/config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=2)

# 3) copy tokenizer
shutil.copy('data/tokenizer.json', f'{EXPORT_DIR}/tokenizer.json')

# ตรวจผล
print("\nไฟล์ที่ export:")
for f in sorted(os.listdir(EXPORT_DIR)):
    size = os.path.getsize(f'{EXPORT_DIR}/{f}')
    print(f"  {f:25} {size/1024/1024:>8.2f} MB")
```

**ขนาดที่ควรเป็น:**
| ไฟล์ | ขนาด |
|---|---|
| `pytorch_model.bin` | ~33 MB |
| `tokenizer.json` | ~0.15 MB |
| `config.json` | ~0.00 MB |

### Cell 6b — Export safetensors (ทางเลือกที่ปลอดภัยกว่า)

```python
from safetensors.torch import save_file

sd = {k: v.contiguous() for k, v in ckpt['model_state_dict'].items()}
save_file(sd, f'{EXPORT_DIR}/model.safetensors')

size = os.path.getsize(f'{EXPORT_DIR}/model.safetensors')
print(f"model.safetensors: {size/1024/1024:.2f} MB")
print("\n💡 ขนาดเท่า .bin แต่ปลอดภัยกว่า (ไม่มี code execution ตอนโหลด)")
```

---

## Cell 7 — Download กลับเครื่อง

```python
import shutil, os
from google.colab import files

shutil.make_archive('/content/guppy_export', 'zip', EXPORT_DIR)
size = os.path.getsize('/content/guppy_export.zip') / 1024 / 1024
print(f"ZIP size: {size:.2f} MB")

files.download('/content/guppy_export.zip')
```

> 💡 **ทางเลือกที่เสถียรกว่า:** เปิด https://drive.google.com แล้วดาวน์โหลดโฟลเดอร์ `guppylm_workshop/export` โดยตรง — ไฟล์อยู่ที่นั่นแล้ว และไม่หายแม้ session ตาย

---

## Cell 8 (ทางเลือก) — Upload ขึ้น HuggingFace Hub

```python
!pip install -q huggingface_hub
from huggingface_hub import login, create_repo, upload_folder

login()   # ใส่ token ที่มี write permission

USERNAME = "your-username"      # ⬅️ แก้เป็นชื่อของตัวเอง
REPO = f"{USERNAME}/my-guppy"

create_repo(REPO, exist_ok=True)
upload_folder(
    folder_path=EXPORT_DIR,
    repo_id=REPO,
    commit_message="GuppyLM from RMUTSV workshop",
)
print(f"อัปโหลดแล้ว: https://huggingface.co/{REPO}")
```

---

## Cell 9 (ทางเลือก) — Resume จาก Checkpoint

ใช้เมื่อ session ตายกลางคัน:

```python
import torch, os

resume_path = f'{CKPT_DIR}/best_model.pt'
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
    print("พบ checkpoint — สามารถ resume ได้")
    print(f"Parameters: {sum(v.numel() for v in ckpt['model_state_dict'].values()):,}")
else:
    print("ไม่พบ checkpoint — ต้องเทรนใหม่")
```

> ⚠️ **ข้อจำกัด:** checkpoint ของ GuppyLM มีแค่ `model_state_dict` และ `config`
> **ไม่มี `optimizer_state_dict` และ `step`** → resume ได้แค่ weights ไม่ต่อ optimizer momentum

---

## Cell 10 (ทางเลือก) — CPU-Friendly Config สำหรับ Plan C

ใช้เมื่อไม่ได้ GPU — เทรนจบใน ~5 นาทีบน CPU (คุณภาพต่ำ แต่เห็น loss ลดจริง)

```python
# แก้ config ให้เล็กลงมาก
CPU_CONFIG = """
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_hidden: int = 256
    max_seq_len: int = 64
"""
print("CPU-friendly config:")
print(CPU_CONFIG)
print("""
TrainConfig:
    batch_size = 16
    max_steps = 300
    warmup_steps = 30
    eval_interval = 50
    save_interval = 100
""")
print("แก้ค่าเหล่านี้ใน guppylm/config.py ก่อนเทรน")
```

---

## 📋 Checklist ก่อนปิด Colab

- [ ] checkpoint อยู่ใน Google Drive แล้ว
- [ ] export ครบ 3 ไฟล์
- [ ] ดาวน์โหลด ZIP ลงเครื่องแล้ว
- [ ] แตก ZIP ตรวจแล้วว่าไฟล์ครบ ขนาดถูกต้อง

---

[⬅️ กลับหน้าหลัก](../../README.md)
