[⬅️ Module 01](01-llm-fundamentals.md) | [หน้าหลัก](../README.md) | [ถัดไป: Model & Training ➡️](03-model-training.md)

---

# 📚 Module 02 — Data & Tokenizer

> ⏱️ **75 นาที** · Lab ปฏิบัติบน Google Colab

## 🎯 Learning Objectives
- โหลดและสำรวจ dataset สำหรับเทรน LLM ได้
- เข้าใจ ChatML format และเหตุผลที่ต้องใช้ template
- เทรน BPE tokenizer ของตัวเองได้
- อธิบายความสำคัญของ data diversity ต่อคุณภาพโมเดล

---

## 2.1 เตรียม Colab Environment

### Cell 1 — ตรวจ GPU

```python
!nvidia-smi
```

### Cell 2 — Clone repository

```python
!git clone https://github.com/arman-bd/guppylm.git
%cd guppylm
!pip install -q tokenizers datasets
```

### Cell 3 — Mount Google Drive (สำคัญมาก!)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
WORKSHOP_DIR = '/content/drive/MyDrive/guppylm_workshop'
CKPT_DIR = f'{WORKSHOP_DIR}/checkpoints'
EXPORT_DIR = f'{WORKSHOP_DIR}/export'

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f"✅ Workshop directory: {WORKSHOP_DIR}")
```

> ⚠️ **ทำไมต้อง mount Drive ตั้งแต่ตอนนี้?**
> พื้นที่ `/content/` ใน Colab เป็น **ephemeral** — หายทั้งหมดเมื่อ runtime รีเซ็ตหรือ session ตาย
> ถ้า save checkpoint ไว้ที่นั่นแล้ว session หลุด = เสียงานทั้งหมด

```mermaid
flowchart LR
    subgraph COLAB["Google Colab VM"]
        E["📁 /content/<br/>❌ Ephemeral<br/>หายเมื่อ runtime reset"]
        D["📁 /content/drive/MyDrive/<br/>✅ Persistent<br/>เก็บถาวรใน Google Drive"]
    end

    T["🔥 Training"] -->|"❌ อย่า save ที่นี่"| E
    T -->|"✅ save ที่นี่"| D
    D --> DL["📥 ดาวน์โหลดได้ภายหลัง<br/>แม้ session ตายไปแล้ว"]

    style E fill:#ffcdd2,stroke:#c62828
    style D fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

---

## 2.2 สำรวจ Dataset

GuppyLM ใช้ **synthetic dataset** ขนาด 60,000 บทสนทนา ครอบคลุม 60 หัวข้อ

### Cell 4 — โหลดและดู dataset

```python
from datasets import load_dataset

ds = load_dataset("arman-bd/guppylm-60k-generic")
print(ds)

# ดูตัวอย่าง 3 รายการ
for i in range(3):
    ex = ds['train'][i]
    print(f"\n--- ตัวอย่าง {i+1} ---")
    print(f"category: {ex['category']}")
    print(f"input:    {ex['input']}")
    print(f"output:   {ex['output']}")
```

**✅ ผลลัพธ์ที่ควรเห็น:**
```
DatasetDict({
    train: Dataset({features: ['input','output','category'], num_rows: 57000})
    test:  Dataset({features: ['input','output','category'], num_rows: 3000})
})
```

### Cell 5 — วิเคราะห์ data diversity

```python
from collections import Counter

outputs = ds['train']['output']
unique = len(set(outputs))
total = len(outputs)

print(f"จำนวนทั้งหมด:      {total:,}")
print(f"จำนวนที่ไม่ซ้ำ:    {unique:,}")
print(f"เปอร์เซ็นต์ unique: {unique/total*100:.1f}%")
print(f"ซ้ำเฉลี่ยต่อคำตอบ:  {total/unique:.1f} ครั้ง")

# หมวดหมู่ที่พบ
print("\nTop 10 categories:")
for cat, n in Counter(ds['train']['category']).most_common(10):
    print(f"  {cat}: {n:,}")
```

### 💡 บทเรียนสำคัญเรื่อง Data Diversity

ผู้สร้าง GuppyLM รายงานในบทความว่าที่ 60,000 samples วัดได้ประมาณ **16,000 unique outputs (~27% uniqueness)** หมายความว่าคำตอบแต่ละแบบปรากฏราว 4 ครั้งโดยเฉลี่ย

เทียบกับเวอร์ชันแรกที่มีเพียง 10 response templates ต่อหัวข้อ ซึ่งทำให้คำตอบแต่ละแบบซ้ำถึง **~400 ครั้ง**

```mermaid
flowchart TD
    subgraph BAD["❌ Data ซ้ำมาก (400 ครั้ง/คำตอบ)"]
        B1["โมเดล 'จำ' คำตอบ"] --> B2["ตอบซ้ำเดิมทุกครั้ง"]
        B2 --> B3["ไม่ generalize<br/>เจอคำถามใหม่ = พัง"]
    end

    subgraph GOOD["✅ Data หลากหลาย (4 ครั้ง/คำตอบ)"]
        G1["โมเดลเรียนรู้ 'pattern'"] --> G2["ผสมผสานสร้างคำตอบใหม่"]
        G2 --> G3["generalize ได้<br/>รับมือคำถามใหม่ได้"]
    end

    style BAD fill:#ffebee,stroke:#c62828
    style GOOD fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

> 🔑 **หลักการ:** ถ้าคำตอบซ้ำมากเกินไป โมเดลจะ **memorize** แทน **learn pattern**
> วิธีแก้: ใช้ **template composition** — สุ่มประกอบส่วนย่อย (วัตถุในตู้, ชนิดอาหาร, กิจกรรม) แทนการเขียน template ตายตัว

---

## 2.3 ChatML Format

โมเดลต้องรู้ว่าส่วนไหนคือคำถามของ user ส่วนไหนคือคำตอบของ assistant จึงต้องมี **template**

```mermaid
flowchart TD
    RAW["📄 Raw data<br/>input: 'hi guppy'<br/>output: 'blub blub'"]
    RAW --> TMPL["🏷️ ChatML Template"]
    TMPL --> FMT["&lt;|im_start|&gt;user<br/>hi guppy&lt;|im_end|&gt;<br/>&lt;|im_start|&gt;assistant<br/>blub blub&lt;|im_end|&gt;"]
    FMT --> TOK["🔤 Tokenize"]
    TOK --> IDS["Token IDs พร้อมเทรน"]

    style TMPL fill:#e3f2fd,stroke:#1976d2
    style FMT fill:#f3e5f5,stroke:#7b1fa2
```

### Special Tokens ของ GuppyLM

| Token | ID | หน้าที่ |
|---|---|---|
| `<pad>` | 0 | เติมให้ sequence ยาวเท่ากันใน batch |
| `<\|im_start\|>` | 1 | BOS — เริ่มข้อความ |
| `<\|im_end\|>` | 2 | EOS — จบข้อความ |
| `<unk>` | 3 | token ที่ไม่รู้จัก |

> 📌 **หมายเหตุ:** GuppyLM **ไม่มี system prompt** โดยตั้งใจ เพราะโมเดล 9M ไม่สามารถ conditionally follow instructions ได้อยู่แล้ว บุคลิกจึงต้องฝังไว้ใน training data ทั้งหมด

---

## 2.4 Lab: เทรน BPE Tokenizer เอง

### Cell 6 — เตรียม corpus

```python
import os
os.makedirs('data', exist_ok=True)

# เขียน corpus สำหรับเทรน tokenizer
with open('data/corpus.txt', 'w', encoding='utf-8') as f:
    for ex in ds['train']:
        f.write(ex['input'] + '\n')
        f.write(ex['output'] + '\n')

print(f"corpus size: {os.path.getsize('data/corpus.txt')/1024/1024:.2f} MB")
```

### Cell 7 — เทรน tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tk = Tokenizer(BPE(unk_token="<unk>"))
tk.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=4096,
    special_tokens=["<pad>", "<|im_start|>", "<|im_end|>", "<unk>"],
)

tk.train(["data/corpus.txt"], trainer)
tk.save("data/tokenizer.json")

print(f"✅ Vocabulary size: {tk.get_vocab_size()}")
```

### Cell 8 — ทดสอบ tokenizer

```python
from tokenizers import Tokenizer
tk = Tokenizer.from_file("data/tokenizer.json")

tests = [
    "hi guppy",
    "are you hungry?",
    "what do you think about quantum physics",
]

for t in tests:
    enc = tk.encode(t)
    print(f"\nข้อความ: {t}")
    print(f"  tokens: {enc.tokens}")
    print(f"  ids:    {enc.ids}")
    print(f"  จำนวน:  {len(enc.ids)} tokens")
    print(f"  decode: {tk.decode(enc.ids)}")
```

**✅ ผลลัพธ์ที่ควรสังเกต:**
- คำที่พบบ่อยในโดเมนปลา (water, food, tank) จะเป็น token เดียว
- คำนอกโดเมน (quantum, physics) จะถูกหั่นเป็นหลาย token
- `decode` แล้วได้ข้อความกลับมาใกล้เคียงเดิม

---

## 2.5 Next-Token Dataset

### Cell 9 — ดูว่า dataset สร้าง x, y อย่างไร

```python
sample = "<|im_start|>user\nhi guppy<|im_end|>"
ids = tk.encode(sample).ids

x = ids[:-1]   # input
y = ids[1:]    # target (เลื่อน 1 ตำแหน่ง)

print(f"ids: {ids}")
print(f"x:   {x}")
print(f"y:   {y}")
print("\nการจับคู่ทำนาย:")
for a, b in zip(x[:8], y[:8]):
    print(f"  {tk.decode([a])!r:15} → ทำนาย → {tk.decode([b])!r}")
```

```mermaid
flowchart LR
    subgraph SEQ["Token sequence"]
        direction LR
        A["10"] --- B["25"] --- C["88"] --- D["42"]
    end

    SEQ --> X["x = [10, 25, 88]"]
    SEQ --> Y["y = [25, 88, 42]"]

    X --> P["10→25 · 25→88 · 88→42"]
    Y --> P

    style X fill:#e3f2fd,stroke:#1976d2
    style Y fill:#ffebee,stroke:#c62828
    style P fill:#e8f5e9,stroke:#2e7d32
```

> 📌 **หมายเหตุเชิงเทคนิคสำหรับผู้สอน:**
> GuppyLM คำนวณ loss จาก **ทุก token ที่ไม่ใช่ padding** (ผ่าน `F.cross_entropy(..., ignore_index=0)` ใน `model.py`) โดย**ไม่ mask ส่วน prompt** ซึ่งต่างจากตำราบางเล่มที่ mask เฉพาะส่วน response ด้วย `-100`
> ควรชี้ให้นักศึกษาเห็นความต่างนี้ — ทั้งสองวิธีใช้ได้ แต่ส่งผลต่อสิ่งที่โมเดลเรียนรู้ต่างกัน

---

## 2.6 เตรียมข้อมูลด้วย script ของ repo

แทนที่จะทำทีละขั้นเอง สามารถใช้ script สำเร็จรูป:

```python
!python -m guppylm.prepare_data
```

**สิ่งที่ script ทำ:**
1. โหลด/สร้าง dataset
2. เทรน BPE tokenizer (vocab 4,096)
3. บันทึก `data/train.jsonl`, `data/test.jsonl`, `data/tokenizer.json`

---

## 🧪 Lab Exercises

ทำ [Lab 1–3 ในหน้า Lab Exercises](07-labs.md#lab-1--เปลี่ยน-vocabulary-size)

---

## 📝 สรุป Module 02

| สิ่งที่ทำ | ผลลัพธ์ |
|---|---|
| Mount Google Drive | มีที่เก็บ checkpoint ถาวร |
| โหลด dataset | เข้าใจโครงสร้าง 60K conversations |
| วิเคราะห์ diversity | เข้าใจว่าทำไม data ซ้ำมากถึงไม่ดี |
| เทรน BPE tokenizer | ได้ `tokenizer.json` vocab 4,096 |
| สร้าง x/y pairs | เข้าใจ next-token prediction |

---

[⬅️ Module 01](01-llm-fundamentals.md) | [หน้าหลัก](../README.md) | [ถัดไป: Model & Training ➡️](03-model-training.md)
