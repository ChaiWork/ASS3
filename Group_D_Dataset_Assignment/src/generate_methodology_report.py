"""
Assignment 3B — Methodology & Model Implementation Report Generator (generate_methodology_report.py)
Group D — STINK3114 Natural Language Processing (UUM)

Generates both:
  1. Editable Microsoft Word Report: `reports/Methodology_Model_Implementation_Group_D.docx`
  2. Submission-Ready PDF Report   : `reports/Methodology_Model_Implementation_Group_D.pdf`

Strictly follows the Assignment 3B Methodology structure (2–4 pages):
  1. Overview of Approach (Method Selection & Justification)
  2. Model / Algorithm Description (Multinomial Logistic Regression & Balanced Weighting)
  3. Pipeline / Workflow Diagram & Explanation
  4. Feature Representation Method (TF-IDF + Emoji-to-Text Tokenization)
  5. Experimental Setup & Preliminary Prototype Verification (E1–E4)
  6. Implementation Details, Reproducibility & GitHub Prototype
  References
"""

import json
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"


def set_cell_background(cell, hex_color: str) -> None:
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=32, bottom=32, left=60, right=60) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_styled_table(doc, headers, rows, col_widths=None, font_size=7.8):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(h_text)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=32, bottom=32, left=60, right=60)
        if col_widths and i < len(col_widths):
            hdr_cells[i].width = Inches(col_widths[i])

    for r_idx, r_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg = "F2F6FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(str(val))
            if r_idx == len(rows) - 1 and headers[0] == "Exp":
                run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(font_size)
            set_cell_background(row_cells[c_idx], bg)
            set_cell_margins(row_cells[c_idx], top=26, bottom=26, left=60, right=60)
            if col_widths and c_idx < len(col_widths):
                row_cells[c_idx].width = Inches(col_widths[c_idx])

    return table


def add_section_heading(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(31, 78, 121)


def add_body_paragraph(doc, text: str, bold_prefix: str = "", space_after: float = 2.8) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.07
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(9.3)
    r_body = p.add_run(text)
    r_body.font.name = "Times New Roman"
    r_body.font.size = Pt(9.3)


def build_methodology_docx(docx_path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "experiment_results_summary.json", "r", encoding="utf-8") as f:
        res_data = json.load(f)

    exps = res_data["experiments"]
    deltas = res_data["factorial_deltas_test_macro_f1"]
    dummy = res_data["majority_class_sanity_baseline"]
    meta = res_data["metadata"]

    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.48)
        section.bottom_margin = Inches(0.48)
        section.left_margin = Inches(0.58)
        section.right_margin = Inches(0.58)

    # Title Block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    r_t = p_title.add_run(
        "Methodology and Model Implementation for Multilingual Social Media Sentiment Analysis:\n"
        "A Controlled 2x2 Evaluation of Emoji Tokenization and Class-Balanced Logistic Regression"
    )
    r_t.bold = True
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(12.0)
    r_t.font.color.rgb = RGBColor(31, 78, 121)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(4)
    r_m = p_meta.add_run(
        "STINK3114 Natural Language Processing (Session I 2026/2027 - A261) | Model Implementation (Methodology) | Group D\n"
        "Chairis A/LPum (303800), Chen Wei Yu (303822), Amanda Soh Yue Ru (303873), Loh Yi Kuan (305590) | "
        "Lecturer: Dr. Muhammad Syafiq bin Mohd Pozi\n"
        "GitHub Repository: https://github.com/ChaiWork/ASS3"
    )
    r_m.italic = True
    r_m.font.name = "Times New Roman"
    r_m.font.size = Pt(8.3)

    # ==========================================================================
    # 1. OVERVIEW OF APPROACH
    # ==========================================================================
    add_section_heading(doc, "1. Overview of Approach (Method Selection & Justification)")
    add_body_paragraph(
        doc,
        "To systematically address Group D's three Research Objectives (RO1: Emoji Retention vs. Removal; "
        "RO2: Class-Imbalance Mitigation; RO3: Combined Experimental Interaction), we adopt a supervised "
        "multi-class text classification framework pairing Term Frequency-Inverse Document Frequency (TF-IDF) "
        "feature extraction with Multinomial Logistic Regression. As established in Assignment 1 (Literature Review) "
        "and Assignment 2 (Research Objectives), prior studies evaluated non-textual emoji retention (Dadure et al., "
        "2025) and class skewness (Eusha et al., 2024; Hande et al., 2021) across disparate model complexities, "
        "making it difficult to separate architectural gains from disciplined data-level and loss-level interventions. "
        "Our methodology explicitly eliminates architectural confounding by locking a single, lightweight linear "
        "classifier across a controlled 2x2 factorial experimental matrix (Experiments E1–E4).",
        bold_prefix="Alignment with Research Problem: ",
    )
    add_body_paragraph(
        doc,
        "Selecting TF-IDF + Logistic Regression is strongly justified by three dataset and literature considerations: "
        "(1) Dataset Scale & Characteristics: Our Assignment 3A corpus contains N = 1,712 short, noisy Malaysian "
        "YouTube comments (median 10 words; 66.41% Bahasa Melayu, 16.41% code-switched Manglish, 4.91% English; "
        "32.54% emoji-bearing; 2.68:1 Neutral-to-Negative skew). At this medium sample scale (1,198 training samples), "
        "regularized linear models avoid the severe overfitting and stochastic instability typical of fine-tuning "
        "100M+ parameter Transformers; (2) Literature Grounding: Advani et al. (2020), Kusampudi et al. (2021), and "
        "Eusha et al. (2024) demonstrated that feature-engineered TF-IDF + Logistic Regression provides a strong, "
        "reproducible CPU-executable baseline for code-mixed social media text; and (3) Direct Interpretability & "
        "Cost-Sensitive Control: Logistic Regression optimizes a convex cross-entropy objective with native inverse-"
        "frequency class weighting (class_weight='balanced') and produces transparent per-class feature coefficients, "
        "allowing direct inspection of how individual emoji text tokens contribute to Positive, Neutral, and Negative "
        "predictions.",
        bold_prefix="Justification from Dataset & Literature: ",
    )

    # ==========================================================================
    # 2. MODEL / ALGORITHM DESCRIPTION
    # ==========================================================================
    add_section_heading(doc, "2. Model / Algorithm Description")
    add_body_paragraph(
        doc,
        "Multinomial Logistic Regression models the posterior probability of each sentiment class k in {Positive, "
        "Neutral, Negative} given a sparse TF-IDF document vector x_i in R^V via the softmax function: "
        "P(y_i = k | x_i) = exp(w_k^T x_i + b_k) / sum_{j=1}^{3} exp(w_j^T x_i + b_j), where w_k and b_k are the "
        "learned weight vector and bias for class k. Parameters are estimated using the deterministic L-BFGS "
        "(Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver with L2 regularization controlled by inverse "
        "penalty C (minimizing 0.5 * ||W||_2^2 + C * sum_{i=1}^{N} alpha_{y_i} * [-log P(y_i | x_i)]).",
        bold_prefix="Multinomial Classification Formulation: ",
    )
    add_body_paragraph(
        doc,
        "Under standard unweighted training (Experiments E1 and E2; class_weight=None), every sample carries equal "
        "weight alpha_k = 1.0, allowing majority Neutral comments (51.40%) to dominate gradient updates and causing "
        "minority Negative recall to collapse. Under cost-sensitive balanced training (Experiments E3 and E4; "
        "class_weight='balanced'), inverse-frequency class weights are computed strictly from the Training split "
        "(N_train = 1,198, K = 3) using alpha_k = N_train / (K * N_train,k), yielding exact weights: Neutral = 0.6483 "
        "(616 samples), Positive = 1.1313 (353 samples), and Negative = 1.7438 (229 samples). Consequently, each "
        "misclassified Negative comment incurs 2.69x the loss penalty of a Neutral comment without fabricating "
        "synthetic oversampled text.",
        bold_prefix="Class-Imbalance Mitigation (RO2): ",
    )

    # ==========================================================================
    # 3. PIPELINE / WORKFLOW DIAGRAM & EXPLANATION
    # ==========================================================================
    add_section_heading(doc, "3. Pipeline / Workflow Diagram and Explanation")
    add_body_paragraph(
        doc,
        "Figure 1 illustrates the end-to-end data flow from the finalized Assignment 3A dataset through dual-track "
        "emoji branching, leakage-free TF-IDF vectorization, controlled Logistic Regression training, and held-out "
        "evaluation. Every comment passes through five deterministic stages: (1) Input Loading of locked stratified "
        "70/15/15 splits (train.csv, validation.csv, test.csv) ensuring identical document indices across all runs; "
        "(2) Dual-Track Branching where E1 and E3 consume cleaned_text_no_emojis (emojis removed) while E2 and E4 "
        "consume cleaned_text_with_emojis transformed via our emoji-to-text-token module (src/emoji_tokenizer.py); "
        "(3) Feature Extraction fitting TfidfVectorizer strictly on X_train and applying .transform() to X_val and "
        "X_test; (4) Model Training fitting LogisticRegression with either class_weight=None or class_weight='balanced' "
        "at fixed random_state=42; and (5) Prediction & Multi-Metric Evaluation outputting 3-class predictions, "
        "probabilities, Macro-F1, Accuracy, Precision, Recall, and Confusion Matrices.",
        space_after=2.0,
    )

    p_fig1 = doc.add_paragraph()
    p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fig1.paragraph_format.space_before = Pt(1)
    p_fig1.paragraph_format.space_after = Pt(1)
    p_fig1.add_run().add_picture(str(FIGURES_DIR / "methodology_pipeline.png"), width=Inches(6.6))
    p_cap1 = doc.add_paragraph()
    p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap1.paragraph_format.space_before = Pt(0)
    p_cap1.paragraph_format.space_after = Pt(4)
    rc1 = p_cap1.add_run(
        "Figure 1: Group D end-to-end NLP methodology pipeline illustrating dual-track emoji handling (E1/E3 vs. E2/E4), "
        "train-only TF-IDF feature representation, and Multinomial Logistic Regression."
    )
    rc1.italic = True
    rc1.font.name = "Times New Roman"
    rc1.font.size = Pt(7.8)

    # ==========================================================================
    # 4. FEATURE REPRESENTATION METHOD
    # ==========================================================================
    add_section_heading(doc, "4. Feature Representation Method (TF-IDF + Emoji Text Tokens)")
    add_body_paragraph(
        doc,
        "Each comment is converted into a sparse numerical vector using sublinear TF-IDF weighting: "
        "TF-IDF(t, d) = [1 + log(tf_{t,d})] * [log((1 + N_train) / (1 + df_t)) + 1], followed by L2 normalization. "
        "Four carefully motivated TF-IDF design choices align with Malaysian social media text: (i) ngram_range=(1, 2) "
        "captures both single polarity words and critical Malay/English negation or complaint bigrams (e.g., 'tak sedap', "
        "'x boleh', 'not bad', 'harga naik') as well as word-emoji bigrams (e.g., 'menipu emoji_face_with_tears_of_joy'); "
        "(ii) stop_words=None avoids destructive stopword filtering that would erase Malay/English negation particles "
        "('tak', 'tidak', 'bukan', 'not', 'no') while IDF naturally downweights ubiquitous neutral connectors ('yang', "
        "'dan', 'the'); (iii) token_pattern=r'(?u)\\b\\w+\\b' preserves the single-character Malaysian Manglish negation "
        "token 'x' (e.g., 'x berbaloi'); and (iv) sublinear_tf=True + min_df=2 dampen repeated spam tokens while "
        "filtering one-off typos, yielding a compact vocabulary of 2,652 features (Track B) and 2,738 features (Track A).",
        bold_prefix="TF-IDF Vectorization Design: ",
    )
    add_body_paragraph(
        doc,
        "A critical technical pitfall in classical NLP pipelines is that scikit-learn's default alphanumeric regex "
        "strips all Unicode emojis and ASCII emoticons before TF-IDF counting—silently collapsing Track A into Track B. "
        "To solve this, src/emoji_tokenizer.py deterministically converts all 145 unique Unicode emojis, regional flag "
        "pairs (e.g., 🇲🇾 -> emoji_flag_malaysia), skin-tone variants, and ASCII emoticons (<3 -> emoticon_heart) into "
        "space-delimited alphanumeric tokens using Python's unicodedata standard (e.g., 😂 -> emoji_face_with_tears_of_joy; "
        "😢 -> emoji_crying_face; 👍 -> emoji_thumbs_up_sign). This adds 87 recurring emoji unigram/bigram features "
        "directly into the TF-IDF matrix for Experiments E2 and E4.",
        bold_prefix="Emoji-to-Text-Token Conversion (Track A): ",
    )

    # ==========================================================================
    # 5. EXPERIMENTAL SETUP & PROTOTYPE VERIFICATION
    # ==========================================================================
    add_section_heading(doc, "5. Experimental Setup and Prototype Verification Results")
    add_body_paragraph(
        doc,
        "All experiments reuse the locked, composite-stratified (label x has_emoji) 70% Train (1,198) / 15% Validation "
        "(257) / 15% Test (257) splits created in Assignment 3A (random_state=42, zero cross-split ID or text leakage). "
        "To utilize the Validation split strictly without peeking at Test data, we evaluated a transparent grid over "
        "ngram_range in {(1,1), (1,2)} and C in {0.5, 1.0, 2.0, 3.0, 5.0} on validation.csv (logged in "
        "results/validation_tuning_log.csv), selecting the single shared configuration (ngram_range=(1, 2), C=5.0; "
        "highest mean Validation Macro-F1 = 0.6116) and locking it across all four conditions in Table 1. Following "
        "Group D's evaluation standard, Macro-F1 is the primary metric because Accuracy alone masks minority-class "
        "collapse (for comparison, a Majority-Class Dummy baseline achieves 0.5136 Accuracy yet collapses to 0.2262 "
        "Macro-F1 and 0.0000 Negative Recall). Accuracy, Macro Precision, Macro Recall, and Per-Class Recall are "
        "reported as complementary metrics.",
        bold_prefix="Splits, Validation Tuning & Metrics: ",
    )

    t1_headers = [
        "Exp", "Emoji Handling", "Class Weight", "Vocab (Emoji)", "Val Macro-F1",
        "Test Macro-F1", "Test Acc", "Macro Prec / Rec", "Neg Recall (Min.)"
    ]
    t1_rows = []
    for eid in ["E1", "E2", "E3", "E4"]:
        e = exps[eid]
        vm = e["validation_metrics"]
        tm = e["test_metrics"]
        t1_rows.append([
            eid,
            e["experiment"]["emoji_handling"],
            e["experiment"]["class_weight_label"],
            f"{e['vocabulary_size']} ({e['emoji_token_features_in_vocab']})",
            f"{vm['macro_f1']:.4f}",
            f"{tm['macro_f1']:.4f}",
            f"{tm['accuracy']:.4f}",
            f"{tm['macro_precision']:.3f} / {tm['macro_recall']:.3f}",
            f"{tm['per_class']['Negative']['recall']:.3f}",
        ])
    add_styled_table(doc, t1_headers, t1_rows, col_widths=[0.42, 1.28, 0.72, 0.84, 0.76, 0.80, 0.62, 1.02, 0.88], font_size=7.6)

    add_body_paragraph(
        doc,
        f"Table 1 and Figures 2–3 confirm that our working prototype runs end-to-end and already exposes compelling "
        f"factorial dynamics across RO1–RO3 on the held-out Test set (n=257): (1) Baseline E1 suffers from severe "
        f"majority-class bias (Neutral Recall = 0.773 vs. minority Negative Recall = 0.400; Macro-F1 = {exps['E1']['test_metrics']['macro_f1']:.4f}); "
        f"(2) Adding emojis alone without class balancing (E2 vs. E1) yields ΔMacro-F1 = {deltas['RO1_emoji_effect_unweighted_E2_minus_E1']:+.4f}, "
        f"because polysemous emojis like 😂 occur heavily in majority Neutral comments; (3) Balanced class weighting alone "
        f"(E3 vs. E1) lifts Macro-F1 by {deltas['RO2_balance_effect_no_emoji_E3_minus_E1']:+.4f} and boosts minority Negative Recall "
        f"from 0.400 to 0.600 (+20.0 pp); and (4) Combining emoji text tokens with balanced weights (E4) achieves the "
        f"highest Test Macro-F1 ({exps['E4']['test_metrics']['macro_f1']:.4f}, {deltas['RO3_combined_gain_E4_minus_E1']:+.4f} over E1, "
        f"interaction term = {deltas['RO3_interaction_term_E4_minus_E3_minus_E2_plus_E1']:+.4f}), highest minority Negative Recall "
        f"(0.620, +22.0 pp over E1), and recovers peak Accuracy (0.6615). Inspecting the learned E4 Logistic Regression "
        f"coefficients (figures/top_emoji_feature_weights.png) verifies that emojis like thumbs_up (+1.41) and "
        f"red_heart (+1.35) anchor Positive predictions while crying_face (+1.83) and the sarcastic bigram "
        f"'menipu emoji_face_with_tears_of_joy' (+0.83) anchor Negative predictions.",
        bold_prefix="Preliminary Prototype Verification: ",
        space_after=2.5,
    )

    # Side-by-side Figures 2 and 3
    fig_tbl = doc.add_table(rows=1, cols=2)
    fig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    specs = [
        (0, FIGURES_DIR / "experiment_metrics_comparison.png", "Figure 2: Held-out Test Macro-F1, Accuracy, Macro Recall, and Minority Negative Recall across E1–E4."),
        (1, FIGURES_DIR / "confusion_matrices_e1_e4.png", "Figure 3: Held-out Test confusion matrices showing recovery of minority Negative recall from E1 (40%) to E4 (62%)."),
    ]
    for c_i, fpath, cap in specs:
        cell = fig_tbl.rows[0].cells[c_i]
        cell.width = Inches(3.65)
        pi = cell.paragraphs[0]
        pi.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pi.paragraph_format.space_before = Pt(1)
        pi.paragraph_format.space_after = Pt(1)
        pi.add_run().add_picture(str(fpath), width=Inches(3.52))
        pc = cell.add_paragraph()
        pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pc.paragraph_format.space_before = Pt(0)
        pc.paragraph_format.space_after = Pt(2)
        rc = pc.add_run(cap)
        rc.italic = True
        rc.font.name = "Times New Roman"
        rc.font.size = Pt(7.5)

    # ==========================================================================
    # 6. IMPLEMENTATION DETAILS & REPRODUCIBILITY
    # ==========================================================================
    add_section_heading(doc, "6. Implementation Details, Reproducibility, and Working GitHub Prototype")
    add_body_paragraph(
        doc,
        f"The complete working system is implemented in Python {meta['python_version']} using scikit-learn "
        f"{meta['sklearn_version']}, pandas, numpy, joblib, and matplotlib, and version-controlled in Group D's GitHub "
        f"repository (https://github.com/ChaiWork/ASS3). Table 2 documents all locked implementation parameters and "
        f"prototype artifacts to guarantee exact bit-for-bit reproducibility on any CPU machine without GPU dependencies.",
        space_after=2.0,
    )

    t2_headers = ["Component", "Module / Artifact Path", "Locked Key Parameters / Reproducibility Specification"]
    t2_rows = [
        ["Emoji Tokenization", "src/emoji_tokenizer.py", "Converts 145+ Unicode emojis, flags (🇲🇾->emoji_flag_malaysia) & emoticons to alphanumeric tokens"],
        ["TF-IDF Vectorizer", "src/train_evaluate_models.py", "ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True, stop_words=None, token_pattern=r'(?u)\\b\\w+\\b'"],
        ["Classifier (E1–E4)", "models/E1..E4_*.joblib", "LogisticRegression(C=5.0, solver='lbfgs', max_iter=1000, random_state=42, class_weight=None|'balanced')"],
        ["Evaluation & Logs", "results/experiment_results_summary.json", "Locked 70/15/15 splits (seed=42); Validation tuning log + Test Macro-F1, Acc, Prec, Rec, CM"],
        ["Working Prototype", "src/predict_prototype.py", "CLI inference tool (`--demo` or `--text \"...\"`) comparing E1–E4 predictions & probabilities live"],
    ]
    add_styled_table(doc, t2_headers, t2_rows, col_widths=[1.25, 2.15, 3.94], font_size=7.6)

    add_body_paragraph(
        doc,
        "Running `python src/predict_prototype.py --demo` demonstrates live multilingual inference across E1–E4. For "
        "example, on the Manglish food review 'Makanan kat bazar ni sedap gila bro, harga pun berbaloi 😍🔥👍', "
        "E1 (Removed+None), E2 (Retained+None), and E3 (Removed+Balanced) all misclassify it as Neutral (P_pos = 0.279, "
        "0.457, 0.352), whereas E4 (Retained+Balanced) combines the learned emoji tokens (emoji_thumbs_up_sign, "
        "emoji_fire) with balanced decision boundaries to accurately predict Positive (P_pos = 0.566).",
        bold_prefix="Live Prototype Demonstration: ",
        space_after=2.0,
    )

    # ==========================================================================
    # REFERENCES
    # ==========================================================================
    add_section_heading(doc, "References")
    refs = [
        "1. Advani, L., Sharma, L., & Mamidi, R. (2020). C1 at SemEval-2020 Task 9: SentiMix: Sentiment analysis for code-mixed social media text using feature engineering. Proceedings of SemEval-2020, 1084–1090.",
        "2. Dadure, P., Dixit, A., Tewatia, K., Paliwal, N., & Malla, A. (2025). Sentiment analysis of Arabic tweets using Large Language Models. Proceedings of AbjadNLP 2025, 88–94.",
        "3. Eusha, A. H., Farsi, S., Hossain, A. I. J., Ahsan, S., & Hoque, M. M. (2024). Sentiment analysis using transformer-based models in code-mixed and transliterated Tamil and Tulu. Proceedings of DravidianLangTech-EACL 2024, 205–211.",
        "4. Hande, A., Priyadharshini, R., & Chakravarthi, B. R. (2021). KanCMD: Kannada code-mixed dataset for sentiment analysis and offensive language detection. Proceedings of RANLP 2021, 450–457.",
        "5. Kusampudi, S. A., Sitaram, S., & Sharma, D. M. (2021). Sentiment analysis in code-mixed Telugu-English text with unsupervised data normalization. Proceedings of DravidianLangTech 2021, 44–50.",
        "6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    ]
    for ref in refs:
        pr = doc.add_paragraph()
        pr.paragraph_format.space_before = Pt(0)
        pr.paragraph_format.space_after = Pt(1.0)
        pr.paragraph_format.line_spacing = 1.02
        rr = pr.add_run(ref)
        rr.font.name = "Times New Roman"
        rr.font.size = Pt(7.8)

    doc.save(str(docx_path))
    print(f"[Methodology Report] Saved editable DOCX: {docx_path}")


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> bool:
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(str(docx_path.resolve()))
    doc.SaveAs(str(pdf_path.resolve()), FileFormat=17)
    doc.Close(False)
    word.Quit()
    print(f"[Methodology Report] Converted DOCX to PDF: {pdf_path}")
    return True


if __name__ == "__main__":
    docx_out = REPORTS_DIR / "Methodology_Model_Implementation_Group_D.docx"
    pdf_out = REPORTS_DIR / "Methodology_Model_Implementation_Group_D.pdf"
    build_methodology_docx(docx_out)
    convert_docx_to_pdf(docx_out, pdf_out)
