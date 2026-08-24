[⬅️ Module 00](00-preparation.md) | [หน้าหลัก](../README.md) | [ถัดไป: Data & Tokenizer ➡️](02-data-tokenizer.md)

---

# 🧠 Module 01 — LLM Fundamentals

> ⏱️ **60 นาที** · บรรยาย + demo

## 🎯 Learning Objectives
เมื่อจบ module นี้ นักศึกษาจะสามารถ:
- อธิบายได้ว่า LLM ทำนาย token ถัดไปอย่างไร
- บอกหน้าที่ของ tokenizer, embedding, attention, sampling ได้
- เข้าใจว่า GuppyLM (8.7M) ต่างจาก GPT-4 อย่างไร — และเหมือนกันตรงไหน

---

## 1.1 LLM ทำอะไรกันแน่?

**คำตอบสั้น ๆ: ทำนาย token ถัดไป**

LLM ไม่ได้ "เข้าใจ" ภาษาแบบมนุษย์ มันเรียนรู้ **การกระจายความน่าจะเป็น** ของ token ถัดไป เมื่อรู้ token ก่อนหน้าทั้งหมด

```mermaid
flowchart LR
    I["'ปลาว่ายอยู่ใน'"] --> M["🧠 LLM"]
    M --> P["ความน่าจะเป็นของ token ถัดไป"]
    P --> O1["'น้ำ' — 62%"]
    P --> O2["'ตู้' — 18%"]
    P --> O3["'ทะเล' — 12%"]
    P --> O4["... — 8%"]
    O1 --> S["🎲 Sampling<br/>เลือก 1 token"]
    O2 --> S
    O3 --> S
    O4 --> S
    S --> R["'ปลาว่ายอยู่ในน้ำ'"]
    R -.->|"วนซ้ำ"| M

    style M fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style S fill:#fff3e0,stroke:#f57c00
```

การทำซ้ำวงจรนี้ทีละ token เรียกว่า **autoregressive generation**

---

## 1.2 Pipeline ทั้งหมดของ LLM

```mermaid
flowchart TD
    subgraph INPUT["1️⃣ Input Processing"]
        T1["ข้อความ<br/>'hi guppy'"] --> T2["🔤 Tokenizer<br/>BPE"]
        T2 --> T3["Token IDs<br/>[1523, 892]"]
    end

    subgraph EMBED["2️⃣ Embedding"]
        T3 --> E1["📊 Token Embedding<br/>ID → vector 384 มิติ"]
        E1 --> E2["➕ Positional Embedding<br/>บอกตำแหน่งใน sequence"]
    end

    subgraph BLOCKS["3️⃣ Transformer Blocks × 6"]
        E2 --> B1["🔍 Multi-Head Attention<br/>token มองหากันเอง"]
        B1 --> B2["📐 LayerNorm + Residual"]
        B2 --> B3["⚡ Feed-Forward Network<br/>384 → 768 → 384, ReLU"]
        B3 --> B4["📐 LayerNorm + Residual"]
        B4 -.->|"ทำซ้ำ 6 รอบ"| B1
    end

    subgraph OUT["4️⃣ Output"]
        B4 --> O1["🎯 LM Head<br/>384 → 4096 logits"]
        O1 --> O2["📈 Softmax<br/>→ probability"]
        O2 --> O3["🎲 Sampling<br/>temperature + top-k"]
        O3 --> O4["Token ถัดไป"]
    end

    style INPUT fill:#e3f2fd,stroke:#1976d2
    style EMBED fill:#f3e5f5,stroke:#7b1fa2
    style BLOCKS fill:#e8f5e9,stroke:#388e3c
    style OUT fill:#fff3e0,stroke:#f57c00
```

---

## 1.3 Tokenizer — สะพานระหว่างข้อความกับตัวเลข

คอมพิวเตอร์ทำงานกับตัวเลข ไม่ใช่ตัวอักษร **tokenizer** คือตัวแปลง

### ทำไมไม่แยกทีละตัวอักษร หรือทีละคำ?

```mermaid
flowchart TD
    Q{"จะหั่นข้อความอย่างไร?"}

    Q --> C["📝 Character-level<br/>h-i-space-g-u-p-p-y"]
    Q --> W["📗 Word-level<br/>hi | guppy"]
    Q --> B["✂️ Subword BPE<br/>hi | gup | py"]

    C --> C1["✅ vocab เล็กมาก<br/>❌ sequence ยาวมาก<br/>❌ ต้องเรียนรู้การสะกดเอง"]
    W --> W1["✅ sequence สั้น<br/>❌ vocab ใหญ่มหาศาล<br/>❌ เจอคำใหม่ = unknown"]
    B --> B1["✅ สมดุลทั้งสองด้าน<br/>✅ จัดการคำใหม่ได้<br/>⭐ ที่ LLM สมัยใหม่ใช้"]

    style B fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style B1 fill:#c8e6c9,stroke:#2e7d32
```

### BPE ทำงานอย่างไร

**Byte-Pair Encoding** เริ่มจากตัวอักษรเดี่ยว แล้ว "รวมคู่ที่พบบ่อยที่สุด" ซ้ำ ๆ จนได้ vocabulary ตามขนาดที่ต้องการ

```mermaid
flowchart TD
    S0["เริ่มต้น: ตัวอักษรเดี่ยว<br/>l, o, w, e, r, n, s, t"]
    S0 --> S1["รอบ 1: 'l'+'o' พบบ่อยสุด<br/>→ สร้าง token 'lo'"]
    S1 --> S2["รอบ 2: 'lo'+'w' พบบ่อยสุด<br/>→ สร้าง token 'low'"]
    S2 --> S3["รอบ 3: 'e'+'s' พบบ่อยสุด<br/>→ สร้าง token 'es'"]
    S3 --> S4["... ทำซ้ำจนครบ 4,096 tokens"]
    S4 --> R["📚 Vocabulary สุดท้าย<br/>ผสมระหว่างตัวอักษร,<br/>ส่วนของคำ, และคำเต็ม"]

    style R fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
```

### เปรียบเทียบขนาด vocabulary

| Model | Vocab size | หมายเหตุ |
|---|---|---|
| **GuppyLM** | 4,096 | domain แคบมาก (เรื่องปลา) จึงพอ |
| GPT-2 | 50,257 | ภาษาอังกฤษทั่วไป |
| LLaMA 2 | 32,000 | multilingual |
| GPT-4 (o200k) | ~200,000 | multilingual + code |

> 💡 **บทเรียนสำคัญ:** GuppyLM ใช้ vocab แค่ 4,096 ได้เพราะ**จำกัด domain ให้แคบมาก** — นี่คือกุญแจที่ทำให้โมเดล 8.7M ให้ผลลัพธ์ที่ดูสมเหตุสมผลได้

---

## 1.4 Attention — หัวใจของ Transformer

**ปัญหา:** เมื่อโมเดลอ่านคำว่า "มัน" ในประโยค "ปลาว่ายในตู้เพราะ**มัน**หิว" — "มัน" หมายถึงอะไร?

**คำตอบ:** attention ให้ token แต่ละตัว "มองไปหา" token อื่นทั้งหมด แล้วให้น้ำหนักว่าตัวไหนสำคัญ

### Query, Key, Value

```mermaid
flowchart LR
    X["Token vector<br/>384 มิติ"] --> QKV["Linear<br/>384 → 1152"]
    QKV --> Q["🔎 Query<br/>'ฉันกำลังหาอะไร'"]
    QKV --> K["🏷️ Key<br/>'ฉันมีอะไรให้'"]
    QKV --> V["📦 Value<br/>'ข้อมูลจริงของฉัน'"]

    Q --> DOT["Q · Kᵀ<br/>÷ √64"]
    K --> DOT
    DOT --> MASK["🚫 Causal Mask<br/>ห้ามมองอนาคต"]
    MASK --> SM["Softmax<br/>→ น้ำหนัก"]
    SM --> MUL["น้ำหนัก × V"]
    V --> MUL
    MUL --> OUT["Output<br/>384 มิติ"]

    style Q fill:#ffebee,stroke:#c62828
    style K fill:#e8eaf6,stroke:#3949ab
    style V fill:#e0f2f1,stroke:#00695c
    style MASK fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

### Causal Mask — ทำไมต้องห้ามมองอนาคต?

เพราะตอน generate จริง โมเดลยังไม่รู้อนาคต ถ้าตอนเทรนให้มองอนาคตได้ โมเดลจะ "โกง" โดยลอกคำตอบ

```mermaid
flowchart TD
    subgraph MATRIX["Attention Mask Matrix (T×T)"]
        direction TB
        M["ตำแหน่ง 1 → มองได้: [1]<br/>ตำแหน่ง 2 → มองได้: [1,2]<br/>ตำแหน่ง 3 → มองได้: [1,2,3]<br/>ตำแหน่ง 4 → มองได้: [1,2,3,4]"]
    end
    MATRIX --> NOTE["🔺 รูปสามเหลี่ยมล่าง (lower triangular)<br/>ส่วนบนถูกตั้งเป็น -∞<br/>→ softmax ให้ค่า 0"]

    style MATRIX fill:#e3f2fd,stroke:#1976d2
    style NOTE fill:#fff3e0,stroke:#f57c00
```

### Multi-Head — ทำไมต้องหลายหัว?

GuppyLM มี 6 heads แต่ละหัวขนาด 64 มิติ (384 ÷ 6 = 64) แต่ละหัวเรียนรู้ "ความสัมพันธ์คนละแบบ" เช่น หัวหนึ่งจับ syntax อีกหัวจับ semantic แล้วนำผลมารวมกัน

---

## 1.5 Next-Token Prediction — วิธีเทรน

```mermaid
flowchart TD
    TEXT["ข้อความต้นฉบับ:<br/>guppy likes water"]
    TEXT --> IDS["Token IDs:<br/>[10, 25, 88, 42]"]

    IDS --> X["📥 Input (x) = ids[:-1]<br/>[10, 25, 88]"]
    IDS --> Y["🎯 Target (y) = ids[1:]<br/>[25, 88, 42]"]

    X --> PAIR["จับคู่กัน:<br/>10 → ทำนาย 25<br/>25 → ทำนาย 88<br/>88 → ทำนาย 42"]
    Y --> PAIR

    PAIR --> LOSS["📉 Cross-Entropy Loss<br/>ทำนายผิดมาก = loss สูง"]
    LOSS --> BP["⬅️ Backpropagation<br/>ปรับ weights"]
    BP -.->|"ทำซ้ำ 10,000 steps"| PAIR

    style X fill:#e3f2fd,stroke:#1976d2
    style Y fill:#ffebee,stroke:#c62828
    style LOSS fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

> 🔑 **นี่คือเหตุผลที่ LLM เทรนได้ด้วยข้อมูลมหาศาลโดยไม่ต้อง label มนุษย์** — ข้อความเองเป็นทั้ง input และ label (self-supervised learning)

**ค่า loss เริ่มต้นที่คาดหวัง:** ถ้าโมเดลเดามั่วสุ่มเท่า ๆ กันทุก token → loss ≈ ln(vocab_size) = ln(4096) ≈ **8.3**
ถ้าเห็นค่าเริ่มต้นใกล้ ๆ นี้แปลว่า initialization ถูกต้อง

---

## 1.6 Sampling — จาก logits เป็นข้อความ

หลังได้ probability distribution แล้วจะเลือก token อย่างไร?

```mermaid
flowchart TD
    L["Logits ดิบ<br/>4,096 ค่า"] --> TEMP["🌡️ หารด้วย Temperature"]

    TEMP --> T1["temp = 0.2<br/>→ distribution แหลม<br/>เลือกตัวที่มั่นใจ"]
    TEMP --> T2["temp = 0.7 ⭐<br/>→ สมดุล<br/>(ค่า default)"]
    TEMP --> T3["temp = 1.5<br/>→ distribution แบน<br/>สุ่มมากขึ้น"]

    T1 --> TOPK["✂️ Top-K Filter<br/>เก็บแค่ K ตัวที่น่าจะเป็นสูงสุด"]
    T2 --> TOPK
    T3 --> TOPK

    TOPK --> SM["Softmax → probability"]
    SM --> MULTI["🎲 torch.multinomial<br/>สุ่มตามน้ำหนัก"]
    MULTI --> TOK["Token ที่เลือก"]
    TOK --> CHECK{"เป็น EOS<br/>หรือครบ max?"}
    CHECK -->|"ไม่"| L
    CHECK -->|"ใช่"| END["✅ จบ"]

    style T2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style MULTI fill:#fff3e0,stroke:#f57c00
```

| Parameter | ค่าต่ำ | ค่าสูง | Default ของ GuppyLM |
|---|---|---|---|
| `temperature` | คาดเดาได้ น่าเบื่อ ซ้ำ | สร้างสรรค์ แต่มั่ว | **0.7** |
| `top_k` | 1 = greedy (ผลเหมือนเดิมทุกครั้ง) | 50+ = หลากหลาย | **50** |

---

## 1.7 GuppyLM vs โมเดลระดับโลก

```mermaid
flowchart LR
    G["🐟 GuppyLM<br/>8.7M"] --> G2["🤖 GPT-2 Small<br/>124M<br/>×14"]
    G2 --> G3["📚 GPT-2 Large<br/>774M<br/>×89"]
    G3 --> G4["🦙 LLaMA 7B<br/>7,000M<br/>×805"]
    G4 --> G5["🌐 GPT-3<br/>175,000M<br/>×20,000"]

    style G fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style G5 fill:#ffcdd2,stroke:#c62828
```

**สิ่งที่เหมือนกัน:**
- ✅ Transformer architecture เดียวกัน
- ✅ Next-token prediction เหมือนกัน
- ✅ Attention mechanism เหมือนกัน
- ✅ Training loop เหมือนกัน

**สิ่งที่ต่าง:** ขนาด, ข้อมูล, และ engineering optimization

> 💬 **ประโยคที่ควรพูดในห้อง:** *"ความต่างระหว่างสิ่งที่เราจะสร้างวันนี้กับ GPT-4 คือ scale — ไม่ใช่แนวคิด ถ้าคุณเข้าใจ GuppyLM คุณเข้าใจแกนกลางของ GPT-4 แล้ว"*

---

## 🎬 Demo: ลองคุยกับ Guppy ก่อน

ให้นักศึกษาเปิด https://arman-bd.github.io/guppylm/ ในมือถือ/laptop

**จุดที่ควรชี้ให้ดู:**
1. โมเดลรัน**ในเบราว์เซอร์ล้วน ๆ** — ไม่มี server ไม่มี API key (ONNX + WebAssembly, ไฟล์ ~10 MB)
2. ตอบเรื่องน้ำ/อาหาร/ตู้ปลาได้ดี แต่ไม่เข้าใจแนวคิดนามธรรม
3. **บุคลิกฝังอยู่ใน weights** — README ของ GuppyLM ระบุตรง ๆ ว่าโมเดล 9M ไม่สามารถ conditionally follow instructions ได้ บุคลิกมาจาก training data ล้วน ๆ

**คำถามชวนคิด:** *"ถ้าเราเปลี่ยนข้อมูลเทรนจากปลาเป็นหุ่นยนต์ โมเดลจะเปลี่ยนไหม? ต้องแก้ architecture ไหม?"*
→ คำตอบ: เปลี่ยนแค่ data ก็พอ — **the model is the data**

---

## 📝 สรุป Module 01

| แนวคิด | หน้าที่ |
|---|---|
| Tokenizer (BPE) | แปลงข้อความ ↔ ตัวเลข |
| Token Embedding | ID → vector ที่มีความหมาย |
| Positional Embedding | บอกตำแหน่งใน sequence |
| Multi-Head Attention | ให้ token มองหากันเอง |
| Causal Mask | ห้ามมองอนาคต |
| FFN | ประมวลผลข้อมูลแต่ละตำแหน่ง |
| LayerNorm + Residual | ทำให้เทรน deep network ได้ |
| LM Head | vector → logits ของทุก token |
| Sampling | logits → token จริง |

---

[⬅️ Module 00](00-preparation.md) | [หน้าหลัก](../README.md) | [ถัดไป: Data & Tokenizer ➡️](02-data-tokenizer.md)
