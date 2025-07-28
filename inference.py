# inference_advanced_lgbm.py

import os
import json
import pickle
import pymupdf
import pandas as pd
import re
from collections import Counter
import numpy as np

# --- Configuration ---
TEST_PDF_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
# --- Point to the new model file ---
MODEL_FILE = "/app/models/xgb_model_full_tuned_multilingual.pkl"
# MODEL_FILE = "./lgbm_model_smote_rfecv.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load the trained model and its artifacts ---
print(f"Loading model and artifacts from {MODEL_FILE}...")
with open(MODEL_FILE, "rb") as f:
    saved_artifacts = pickle.load(f)

model = saved_artifacts["model"]
# --- KEY CHANGE: Load the list of features the model was trained on ---
selected_features = saved_artifacts["features"]
classes = saved_artifacts["classes"]
label_encoder = saved_artifacts["label_encoder"]

print(f"Model loaded. Expecting {len(selected_features)} features.")

# --- Feature Generation Functions (unchanged) ---

NUMBERING_RE = re.compile(
    r'^\s*(?:Chapter|Section|Part|第|章)\s*\d+|'  # English/Japanese Chapter/Section
    r'^\s*(?:[IVXLCDM]+|[A-Za-z]|\d+|[０-９]+|[一二三四五六七八九十百千])[\.)、]\s+|'  # Roman, Alpha, Numeric, Full-width, CJK numerals
    r'^\s*\d+\s+\w+'
)

def correct_heading_hierarchy(outline: list) -> list:
    """
    Corrects the hierarchy of a list of heading dicts.

    Rules:
      1. The very first heading is always H1.
      2. You can never jump more than one level deeper at once.
      3. You may jump up by any number of levels.
    """
    if not outline:
        return []

    corrected = []
    last_level = 1

    for idx, item in enumerate(outline):
        # parse the numeric part of e.g. "H3"
        lvl_str = item.get("level", "")
        try:
            curr = int(lvl_str.lstrip("H"))
        except (ValueError, IndexError):
            curr = last_level

        # force the first heading to level 1
        if idx == 0:
            curr = 1
        else:
            # cap deeper jumps to at most +1
            if curr > last_level + 1:
                curr = last_level + 1
            # upward jumps (curr < last_level) are OK

        item["level"] = f"H{curr}"
        corrected.append(item)
        last_level = curr

    return corrected

def get_font_style(flags: int) -> tuple[bool, bool, bool, bool]:
    """Decodes bold, italic, serif, mono from the PyMuPDF flags integer."""
    is_bold   = bool(flags & 16)
    is_italic = bool(flags &  2)
    is_serif  = bool(flags &  4)
    is_mono   = bool(flags &  8)
    return is_bold, is_italic, is_serif, is_mono


def generate_features_for_page(page: pymupdf.Page, page_number: int) -> list[dict]:
    """Generates a list of language-agnostic feature vectors for each line of text."""
    features = []
    page_dict = page.get_text("dict")
    w, h = page.rect.width, page.rect.height

    # Compute modal font size for the page (language-agnostic)
    font_sizes = [
        span['size']
        for blk in page_dict.get('blocks', []) if blk['type']==0
        for line in blk['lines']
        for span in line['spans']
    ]
    modal_fs = float(Counter(font_sizes).most_common(1)[0][0]) if font_sizes else 1.0
    max_fs   = max(font_sizes) if font_sizes else modal_fs

    raw_lines = []
    for blk_idx, blk in enumerate(page_dict.get('blocks', [])):
        if blk['type'] != 0:
            continue
        for ln_idx, line in enumerate(blk['lines']):
            text = "".join(span['text'] for span in line['spans']).strip()
            if not text:
                continue
            raw_lines.append({
                'page_num': page_number,
                'block_idx': blk_idx,
                'line_idx':  ln_idx,
                'spans':     line['spans'],
                'bbox':      line['bbox'],
                'text':      text,
            })

    total_lines = len(raw_lines)

    # Precompute font size ranks and name IDs (language-agnostic)
    unique_font_sizes = sorted({max(span['size'] for span in L['spans']) for L in raw_lines}, reverse=True)
    font_size_ranks = {size: rank + 1 for rank, size in enumerate(unique_font_sizes)}

    unique_font_names = list({span.get('font', '') for L in raw_lines for span in L['spans']})
    font_name_ids = {name: i for i, name in enumerate(unique_font_names)}

    for idx, line in enumerate(raw_lines):
        spans = line['spans']
        dom = max(spans, key=lambda s: s['size'])
        is_bold, is_italic, is_serif, is_mono = get_font_style(dom.get('flags', 0))
        x0, y0, x1, y1 = line['bbox']
        prev_line = raw_lines[idx - 1] if idx > 0 else None
        next_line = raw_lines[idx + 1] if idx < total_lines - 1 else None
        fs = dom['size']

        space_above = (y0 - prev_line['bbox'][3]) if prev_line else 0.0
        space_below = (next_line['bbox'][1] - y1) if next_line else 0.0

        # --- FEATURE SET FOCUSED ON LANGUAGE-AGNOSTIC PROPERTIES ---
        rec = {
            'line_id':               f"page_{page_number}_line_{idx+1}",
            'text':                  line['text'],
            'page':                  page_number,
            
            # Positional Features (Universal)
            'normalized_x0':         x0 / w,
            'normalized_y0':         y0 / h,
            'normalized_x1':         x1 / w,
            'normalized_y1':         y1 / h,
            'indentation':           x0,
            'rel_page_position':     y0 / h,
            'line_height':           (y1 - y0) / h,
            'line_width_ratio':      ((x1 - x0) / w if w else 0),
            'is_centered':           abs(((x0 + x1) / 2) - w / 2) < (w * 0.02),
            
            # Spacing Features (Universal)
            'spacing_top':           (space_above / h),
            'spacing_bottom':        (space_below / h),
            'space_above':           space_above,
            'space_below':           space_below,
            
            # Font Features (Universal)
            'font_size':             fs,
            'rel_font_size':         fs / modal_fs,
            'is_largest_on_page':    fs == max_fs,
            'font_size_rank':        font_size_ranks.get(fs, -1),
            'font_name_id':          font_name_ids.get(dom.get('font', ''), -1),
            'is_bold':               is_bold,
            'is_italic':             is_italic,
            'is_serif':              is_serif,
            'is_mono':               is_mono,
            
            # Contextual Features (Universal)
            'same_block_as_prev':    (line['block_idx'] == prev_line['block_idx']) if prev_line else False,
            'font_size_prev':        max(span['size'] for span in prev_line['spans']) if prev_line else 0,
            'font_size_next':        max(span['size'] for span in next_line['spans']) if next_line else 0,
            'page_is_first':         page_number == 1,
            'total_lines_on_page':   total_lines,
            
            # Text-based Features (Simplified for Multilingual)
            'char_count':            len(line['text']),
            'digit_ratio':           (sum(1 for c in line['text'] if c.isdigit()) / max(1, len(line['text']))),
            'starts_with_numbering': bool(NUMBERING_RE.match(line['text'])),
        }
        
        # --- REMOVED FEATURES THAT ARE NOT LANGUAGE-AGNOSTIC ---
        # 'word_count', 'avg_word_length', 'uppercase_ratio', 'is_all_caps',
        # 'colon_count', 'ends_with_period', 'word_count_prev', 'word_count_next'
        
        features.append(rec)

    return features

# --- Iterate over test PDFs ---
for pdf_filename in sorted(os.listdir(TEST_PDF_DIR)):
    if not pdf_filename.lower().endswith(".pdf"):
        continue

    print(f"\nProcessing '{pdf_filename}'...")
    pdf_path = os.path.join(TEST_PDF_DIR, pdf_filename)
    doc = pymupdf.open(pdf_path)
    
    # Generate features for all pages
    all_rows = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        all_rows.extend(generate_features_for_page(page, page_num + 1))

    if not all_rows:
        print("  No text found in PDF.")
        continue

    # Create DataFrame from features
    df = pd.DataFrame(all_rows)
    X_inference = df.drop(columns=["line_id", "text"], errors="ignore")

    # --- KEY CHANGE: Align columns with the selected features from training ---
    # This ensures the model gets the exact features it was trained on.
    X_inference = X_inference.reindex(columns=selected_features, fill_value=0)

    # Predict probabilities and get the predicted class index
    predictions = model.predict(X_inference)
    # The output of LGBMClassifier.predict is already the class index
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Build title + outline JSON
    title = next((text for text, label in zip(df["text"], predicted_labels) if label == "Title"), df["text"].iloc[0])
    outline = []
    for line_id, text, label in zip(df["line_id"], df["text"], predicted_labels):
        if label in ("H1", "H2", "H3", "H4"):
            page_num = int(line_id.split("_")[1])
            outline.append({"level": label, "text": text, "page": page_num-1})

    outline = correct_heading_hierarchy(outline)

    output_data = {"title": title, "outline": outline}
    
    # Save the output JSON
    output_filename = f"{os.path.splitext(pdf_filename)[0]}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"  -> Wrote {output_path} with {len(outline)} headings found.")

print("\n✅ All test PDFs processed.")
