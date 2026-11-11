# Group D — Sentiment Annotation Guideline & Adjudication Protocol
**Course**: STINK3114 Natural Language Processing (Session I 2026/2027 - A261)  
**Group**: Group D (Chairis A/LPum, Chen Wei Yu, Amanda Soh Yue Ru, Loh Yi Kuan)  
**Lecturer**: Dr. Muhammad Syafiq bin Mohd Pozi

---

## 1. Purpose and Scope
This guideline governs the three-class sentiment annotation (`Positive`, `Negative`, `Neutral`) of multilingual Malaysian social media comments (English, Bahasa Melayu, and code-switched Malaysian Manglish / *Bahasa Rojak*) collected from YouTube discussion threads. To ensure methodological validity for **Research Objective 1 (RO1)**, **Research Objective 2 (RO2)**, and **Research Objective 3 (RO3)**, annotators must evaluate the holistic semantic and pragmatic meaning of each comment—**never assigning a sentiment label based solely on the presence of an emoji**.

---

## 2. Sentiment Class Definitions

### 2.1 Positive
Expresses satisfaction, happiness, approval, appreciation, praise, support, sincere gratitude, or constructive optimism/well-wishes (*du'a*).
- **Typical Linguistic Cues**: *terbaik, sedap, mantap, padu, sempoi, berbaloi, murah, kemas, suka, puas hati, alhamdulillah, syukur, terima kasih, tahniah, semoga dipermudahkan, worth it, love, great, awesome, good job*.
- **Authentic Corpus Examples**:
  - `GD_0005`: *"I from (san marino)my nama Mario...terbaik Malaysia..😊"* → **Positive**
  - `GD_0010`: *"God bless Malaysia 🇲🇾🥰"* → **Positive**
  - `GD_0041`: *"Terbaiklah😊Salam dari Sarawak😊"* → **Positive**

### 2.2 Negative
Expresses dissatisfaction, frustration, anger, disappointment, criticism, mockery/sarcasm, complaint about defects/service failures, or pessimism/distress.
- **Typical Linguistic Cues**: *teruk, buruk, mahal, lambat, rosak, jem, sesak, masalah, susah, scam, tipu, munafik, songlap, rasuah, zalim, bodoh, koyak, indah khabar dari rupa, cekik darah, tak boleh pakai, buang masa, bad, worst, slow, waste*.
- **Authentic Corpus Examples**:
  - `GD_0017`: *"Tahun 2025 tahun derita😢"* → **Negative**
  - `GD_0028`: *"Songlap duit rakyat nadi utama politik 😂"* → **Negative** (Note: The laughing emoji `😂` expresses cynical mockery of political corruption; lexical complaint governs.)
  - `GD_0058`: *"Anwar dah tak ada rasa Malu sebab memang tak ada maruah sebab kuat menipu 😂😂😂😂😂"* → **Negative**

### 2.3 Neutral
Expresses factual statements, objective information, genuine information-seeking questions, timestamps, neutral observations, or mixed/ambiguous remarks without a dominant positive or negative evaluative stance.
- **Typical Linguistic Cues**: *berapa harga, kat mana, bila, macam mana, siapa, biasa je, sekadar bertanya*.
- **Authentic Corpus Examples**:
  - `GD_0046`: *"Kenapa tarik balik tok.mat?"* → **Neutral** (Genuine question)
  - `GD_0087`: *"Menteri Luar Negeri? Bukannya, Menteri Luar Negara? 😂😂"* → **Neutral** (Terminological clarification question with lighthearted emoji; not expressing positive/negative sentiment)

---

## 3. Decision Rules for Complex Social Media Phenomena

| Phenomenon | Decision Rule | Example & Assigned Label |
|---|---|---|
| **1. Emojis & Emoticons** | Emojis act as **contextual modifiers**, never standalone label triggers. When an emoji conflicts with explicit lexical polarity (e.g., `😂` or `🤣` attached to a complaint or insult), the **pragmatic intent of the text** takes precedence. | *"Songlap duit rakyat nadi utama politik 😂"* → **Negative** (not Positive) |
| **2. Sarcasm & Irony** | Surface praise words (*tahniah, bagus, hebat, membanggakan*) paired with explicit failure, defect, or subjugation (*rosak, jem, lambat, mahal, dijajah, scam*) are annotated as **Negative**. | *"TAHUN MEMBANGGAKAN, DENGAN KEMBALI JADI JAJAHAN AMERIKA SYARIKAT..."* → **Negative** |
| **3. Negation** | Windowed negation markers (*tak, tidak, bukan, jangan, x, xde, takde, tiada, belum, not, never, no*) invert positive adjectives to **Negative** (*tak sedap*, *tak berbaloi*, *bukan terbaik*). Idiomatic double-negations (*takde masalah*, *tiada masalah*, *not bad*, *tak mengecewakan*) resolve to **Positive**. | *"Makanan dia tak sedap langsung"* → **Negative**; *"Setakat ni guna RFID takde masalah"* → **Positive** |
| **4. Contrastive Clauses (`tapi` / `but`)** | In compound sentences joined by contrastive conjunctions (*tapi, tetapi, cuma, namun, but, however*), the **post-conjunction clause** carries the dominant pragmatic sentiment. | *"Rupa nampak cantik tapi rasa tawar dan mahal"* → **Negative** |
| **5. Repeated Characters & Slang** | Expressive character elongation (*sedapppp*, *terbaikk*, *mahallll*, *jemmmm*) and Malaysian slang (*padu, mantap, koyak, barai, hampeh, cekik darah*) are normalized to root forms during interpretation while preserving their intensifying force. | *"Mamak ni memang paduuuuu gila"* → **Positive** |
| **6. Malay-English Code-Switching (Manglish)** | Evaluate bilingual predicates jointly. Notice that *"gila"* immediately preceding or following an adjective (*sedap gila*, *gila best*, *mahal gila*) functions as an **intensifier** ("very/extremely"), not a literal negative insult ("crazy"). | *"Nasi lemak dia sedap gila bro"* → **Positive** |

---

## 4. Dual-Annotator Protocol & Adjudication Summary
Every comment in the final dataset ($N = 1,712$) underwent two independent annotation passes (`ANN_01` Lexical-Constructional & Morphological Annotator and `ANN_02` Pragmatic-Discourse & Clause-Segmentation Annotator), followed by systematic adjudication (`ANN_GROUP_D_ADJUDICATED`) for all disagreements:
- **Raw Inter-Annotator Agreement**: $94.63\%$ ($1,620 / 1,712$ exact matches).
- **Cohen's Kappa ($\kappa$)**: $\kappa = 0.9132$ (*Almost Perfect Agreement* per Landis & Koch, 1977).
- **Adjudicated Disagreements**: $92$ records (logged in `logs/annotation_disagreements.csv`).

