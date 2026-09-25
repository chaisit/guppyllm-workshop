[⬅️ กลับหน้าหลัก](../../README.md)

---

# 📓 Colab / Jupyter Cells พร้อมใช้

> คัดลอกทีละ cell ไปวางใน Google Colab **หรือ** Jupyter Notebook บนเครื่องตัวเอง ตามลำดับ
> Cell ชุดนี้ออกแบบให้รันได้ทั้งสองสภาพแวดล้อมด้วยโค้ดชุดเดียว — Cell 1 จะตรวจเองว่าอยู่บน Colab หรือ local

## 🔢 ลำดับการรัน

```mermaid
flowchart LR
    C1["1️⃣ Setup +<br/>ตรวจ environment"] --> C2["2️⃣ กำหนดที่เก็บ<br/>(Mount Drive / local)"]
    C2 --> C3["3️⃣ Data"]
    C3 --> C4["4️⃣ Train<br/>(in-process)"]
    C4 --> C5["5️⃣ Test"]
    C5 --> C6["6️⃣ Export"]
    C6 --> C7["7️⃣ Download"]

    style C2 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style C4 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C6 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

> ⚠️ **บน Colab: Cell 2 (Mount Drive) ห้ามข้าม** — ถ้าข้าม checkpoint จะหายเมื่อ session ตาย
> 💻 **บน Jupyter local:** Cell 2 จะข้าม mount ให้อัตโนมัติ และเก็บ checkpoint ลงโฟลเดอร์ในเครื่องแทน

> 🔑 **จุดสำคัญที่สุดของชุด cell นี้:** เรารัน `guppylm.train.train()` **ในโปรเซสเดียวกับ notebook** (ไม่ใช่ `!python -m guppylm.train` ที่เป็น subprocess) จึง override `output_dir`/`data_dir` ให้ชี้ไปที่ Drive/โฟลเดอร์ในเครื่องได้จริง — ดูเหตุผลละเอียดใน [Module 03 §3.5](../../docs/03-model-training.md#35-lab-เทรนจริง)

---

## Cell 1 — Setup, ตรวจ environment และ GPU

```python
import os, sys

# ── ตรวจว่าอยู่บน Colab หรือ Jupyter local ──
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print(f"Environment: {'Google Colab' if IN_COLAB else 'Jupyter local'}")

# ── Clone repository (ถ้ายังไม่มี) ──
if not os.path.isdir('guppylm'):
    !git clone https://github.com/arman-bd/guppylm.git

# ให้ import guppylm.* ได้ โดยไม่ต้อง cd เข้า repo
REPO_DIR = os.path.abspath('guppylm')
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ── ติดตั้ง dependencies ──
# บน Colab มี torch อยู่แล้ว; บนเครื่อง local ให้ติดตั้ง torch เอง (ดู code/local/README.md)
!pip install -q tokenizers datasets safetensors

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

> 📌 **ทำไมไม่ `%cd guppylm`?** เพราะเราจะ `import guppylm.train` แบบ package แล้วรัน `train()` ในโปรเซสนี้เลย
> การเพิ่ม repo เข้า `sys.path` แทนการ `cd` ทำให้ working directory ยังเป็นที่ที่เราคุมได้ (สำคัญตอนกำหนดที่เก็บ `data/` และ checkpoint)

---

## Cell 2 — กำหนดที่เก็บ checkpoint ⚠️ สำคัญที่สุด

> บน **Colab** จะ mount Google Drive เพื่อกัน checkpoint หายเมื่อ session ตาย
> บน **Jupyter local** จะเก็บลงโฟลเดอร์ในเครื่องแทน (ไม่ต้อง mount อะไร)

```python
import os

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    WORKSHOP_DIR = '/content/drive/MyDrive/guppylm_workshop'
else:
    # เครื่องตัวเอง — เก็บไว้ในโฟลเดอร์ปัจจุบัน
    WORKSHOP_DIR = os.path.abspath('guppylm_workshop')

DATA_DIR   = os.path.join(WORKSHOP_DIR, 'data')
CKPT_DIR   = os.path.join(WORKSHOP_DIR, 'checkpoints')
EXPORT_DIR = os.path.join(WORKSHOP_DIR, 'export')

for d in (DATA_DIR, CKPT_DIR, EXPORT_DIR):
    os.makedirs(d, exist_ok=True)

print(f"Workshop dir: {WORKSHOP_DIR}")
print(f"Data:         {DATA_DIR}")
print(f"Checkpoints:  {CKPT_DIR}")
print(f"Export:       {EXPORT_DIR}")

if IN_COLAB:
    print("\n>>> ตรวจว่า path ขึ้นต้นด้วย /content/drive/ แล้วยกมือ <<<")
```

> 🔴 **บน Colab: ถ้า path ไม่ขึ้นต้นด้วย `/content/drive/` แปลว่า mount ไม่สำเร็จ — อย่าเทรนต่อ**

---

## Cell 3 — เตรียม Data และ Tokenizer

```python
import os

# ⚠️ สำคัญ: ล็อก working directory เป็น WORKSHOP_DIR ก่อนเรียก prepare()
# generate_dataset ของ repo เขียน data/ แบบ relative ต่อ cwd และ train() ก็อ่าน data/ แบบ relative
# ต้องเป็น cwd เดียวกันทั้งตอนเตรียม data และตอนเทรน ไม่งั้นเทรนจะเจอ FileNotFoundError
os.chdir(WORKSHOP_DIR)
print("cwd =", os.getcwd())

# เรียก prepare() แบบ in-process (ไม่ใช่ !python -m) เพื่อให้เป็นแนวเดียวกับตอนเทรน
from guppylm.prepare_data import prepare
prepare()   # สร้าง data/train.jsonl, data/eval.jsonl, data/tokenizer.json (ใต้ WORKSHOP_DIR)

# ตรวจผลลัพธ์
for f in sorted(os.listdir('data')):
    size = os.path.getsize(f'data/{f}')
    print(f"{f:25} {size/1024:>10,.1f} KB")
assert os.path.exists('data/train.jsonl') and os.path.exists('data/tokenizer.json')
```

> 📌 **สังเกตชื่อไฟล์:** repo สร้าง `train.jsonl` และ **`eval.jsonl`** (ไม่ใช่ `test.jsonl`) — `train.py` โหลด `data/eval.jsonl` มาใช้วัด eval loss
> 🔤 **tokenizer ที่ได้:** BPE แบบ **ByteLevel** vocab 4,096 special tokens 3 ตัว (`<pad>`, `<|im_start|>`, `<|im_end|>`) ตามที่ repo กำหนดจริง
> 📁 **ทำไม `os.chdir(WORKSHOP_DIR)`?** ทั้ง `prepare()` และ `train()` ใช้ path `data/` แบบ relative ต่อ cwd การล็อก cwd ไว้ที่เดียวกันตั้งแต่ตอนนี้กัน `FileNotFoundError: data/train.jsonl` ตอนเทรน (บน Colab ยังได้ persist `data/` ลง Drive ด้วย)

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

### Cell 4a — ตรวจ config ก่อนเทรน

```python
from guppylm.config import GuppyConfig, TrainConfig
from guppylm.model import GuppyLM

mc = GuppyConfig()
tc = TrainConfig()

print("=== Model Config ===")
for k, v in vars(mc).items():
    print(f"  {k:15} = {v}")

print("\n=== Train Config (ค่า default ของ repo) ===")
for k, v in vars(tc).items():
    print(f"  {k:15} = {v}")

m = GuppyLM(mc)
print(f"\n{m.param_summary()}")
del m
```

### Cell 4b — ตั้งที่เก็บ checkpoint ให้ถูกต้อง (หัวใจของการแก้ปัญหา)

> ⚠️ **ทำไมต้องมี cell นี้?**
> `guppylm/train.py` เรียก `TrainConfig()` เองภายในฟังก์ชัน `train()` และ `TrainConfig.output_dir` ถูก **hardcode** เป็น `"checkpoints"`
> ถ้าเรารันด้วย `!python -m guppylm.train` มันจะเป็น **subprocess แยกออกไป** — ตัวแปร `CKPT_DIR` ที่เราตั้งไว้ในโน้ตบุ๊กจะ **ไม่มีผลใด ๆ** และ checkpoint จะตกไปอยู่ใน `checkpoints/` ของ session (หายเมื่อ session ตาย)
>
> ทางแก้ที่ถูกต้อง: รัน `train()` **ในโปรเซสเดียวกับโน้ตบุ๊ก** แล้ว override `TrainConfig` ให้ `output_dir` ชี้ไป `CKPT_DIR` ก่อนเรียก

```python
import os
import guppylm.train as gtrain
from guppylm.config import TrainConfig

# ล็อก cwd ให้ตรงกับตอนรัน prepare() (Cell 3) — train.py อ่าน data/ แบบ relative ต่อ cwd
os.chdir(WORKSHOP_DIR)
assert os.path.exists('data/train.jsonl'), \
    "ไม่พบ data/train.jsonl — ต้องรัน Cell 3 (prepare) จาก cwd เดียวกันนี้ก่อน"

# แทนที่ TrainConfig ที่ train.py ใช้ ด้วยเวอร์ชันที่ output_dir ชี้ไป CKPT_DIR
# (data_dir คง "data" ตามเดิม — ตอนนี้ cwd = WORKSHOP_DIR จึงหมายถึง WORKSHOP_DIR/data)
_BaseTrainConfig = TrainConfig
def _patched_train_config():
    return _BaseTrainConfig(output_dir=CKPT_DIR)

gtrain.TrainConfig = _patched_train_config

# ยืนยันว่า override ติดจริง
print("cwd      =", os.getcwd())
print("output_dir ที่จะใช้จริง:", gtrain.TrainConfig().output_dir)
assert gtrain.TrainConfig().output_dir == CKPT_DIR
```

### Cell 4c — เริ่มเทรน (in-process)

```python
# 🚀 เริ่มเทรน — ใช้เวลา ~5 นาทีบน T4 ที่ว่าง
# เรียก train() ตรง ๆ (ไม่ใช่ !python -m) เพื่อให้ override ข้างบนมีผล
gtrain.train()
```

**สิ่งที่ควรเห็น:**
- loss เริ่มต้นราว **8.3** (= ln(4096))
- loss ลดลงเรื่อย ๆ
- checkpoint ถูกบันทึกลง `CKPT_DIR` (บน Drive ถ้าเป็น Colab) เป็นระยะ

```python
# ✅ ตรวจว่า checkpoint ไปอยู่ที่ถูกที่จริง
import os
print("ไฟล์ใน CKPT_DIR:")
for f in sorted(os.listdir(CKPT_DIR)):
    size = os.path.getsize(os.path.join(CKPT_DIR, f))
    print(f"  {f:20} {size/1024/1024:8.2f} MB")
assert os.path.exists(os.path.join(CKPT_DIR, 'best_model.pt')), \
    "ไม่พบ best_model.pt ใน CKPT_DIR — ตรวจว่ารัน Cell 4b แล้ว"
print("\n✅ checkpoint อยู่ใน CKPT_DIR เรียบร้อย")
```

---

## Cell 5 — ทดสอบโมเดล (ยังอยู่ในโน้ตบุ๊ก)

```python
from guppylm.inference import GuppyInference

engine = GuppyInference(
    checkpoint_path=os.path.join(CKPT_DIR, 'best_model.pt'),
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

> 💻 **บน Jupyter local ไม่ต้องทำ cell นี้** — ไฟล์อยู่ใน `EXPORT_DIR` บนเครื่องคุณอยู่แล้ว

```python
import shutil, os

zip_base = os.path.join(WORKSHOP_DIR, 'guppy_export')
shutil.make_archive(zip_base, 'zip', EXPORT_DIR)
zip_path = zip_base + '.zip'
size = os.path.getsize(zip_path) / 1024 / 1024
print(f"ZIP: {zip_path} ({size:.2f} MB)")

if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
else:
    print("อยู่บนเครื่อง local แล้ว — เปิดไฟล์ ZIP ได้จากโฟลเดอร์ข้างบนโดยตรง")
```

> 💡 **บน Colab ทางเลือกที่เสถียรกว่า:** เปิด https://drive.google.com แล้วดาวน์โหลดโฟลเดอร์ `guppylm_workshop/export` โดยตรง — ไฟล์อยู่ที่นั่นแล้ว และไม่หายแม้ session ตาย

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

resume_path = os.path.join(CKPT_DIR, 'best_model.pt')
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
    print("พบ checkpoint — สามารถ resume ได้")
    print(f"Keys:       {list(ckpt.keys())}")
    print(f"Step:       {ckpt.get('step')}")
    print(f"Parameters: {sum(v.numel() for v in ckpt['model_state_dict'].values()):,}")
else:
    print("ไม่พบ checkpoint — ต้องเทรนใหม่")
```

> ⚠️ **ข้อจำกัด:** `best_model.pt` ของ GuppyLM บันทึก `step`, `model_state_dict`, `config`, `eval_loss`
> แต่ **ไม่มี `optimizer_state_dict`** → resume ได้แค่ weights ไม่ต่อ optimizer momentum และ LR schedule

---

## Cell 10 (ทางเลือก) — CPU-Friendly Config สำหรับ Plan C

ใช้เมื่อไม่ได้ GPU (หรือรันบน Jupyter local ที่ไม่มี GPU) — เทรนจบใน ~5 นาทีบน CPU (คุณภาพต่ำ แต่เห็น loss ลดจริง)

> ใช้แนวเดียวกับ Cell 4b: **override config แบบ in-process** ไม่ต้องแก้ `config.py`
> รัน cell นี้ **แทน Cell 4a–4c** เมื่ออยู่บน CPU

```python
import guppylm.train as gtrain
from guppylm.config import GuppyConfig, TrainConfig

# โมเดลเล็กลงมากให้เทรนไหวบน CPU
def _cpu_model_config():
    return GuppyConfig(d_model=128, n_layers=2, n_heads=4,
                       ffn_hidden=256, max_seq_len=64)

# เทรนสั้นลง + ยังชี้ output_dir ไป CKPT_DIR เหมือนเดิม
def _cpu_train_config():
    return TrainConfig(output_dir=CKPT_DIR, batch_size=16, max_steps=300,
                       warmup_steps=30, eval_interval=50, save_interval=100,
                       device="cpu")

gtrain.GuppyConfig = _cpu_model_config
gtrain.TrainConfig = _cpu_train_config

print("CPU config พร้อม — เริ่มเทรนด้วย gtrain.train()")
gtrain.train()
```

> 📌 `train.py` import ทั้ง `GuppyConfig` และ `TrainConfig` ไว้ใน namespace ของตัวเอง เราจึง patch ที่ `gtrain.*` ทั้งคู่ให้มีผลกับ `train()` ที่กำลังจะเรียก

---

## 📋 Checklist ก่อนปิด Colab / จบขั้นเทรน

- [ ] รัน Cell 4b แล้ว (`gtrain.TrainConfig().output_dir` == `CKPT_DIR`)
- [ ] checkpoint อยู่ใน `CKPT_DIR` แล้ว (บน Colab = Google Drive)
- [ ] export ครบ 3 ไฟล์
- [ ] (Colab) ดาวน์โหลด ZIP ลงเครื่องแล้ว
- [ ] แตก ZIP ตรวจแล้วว่าไฟล์ครบ ขนาดถูกต้อง

---

[⬅️ กลับหน้าหลัก](../../README.md)
