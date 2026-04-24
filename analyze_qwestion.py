"""
EgoCross サブタスク統合分析スクリプト（統計情報強化版）。
ユニークな質問パターン数（types）と、そのサブタスクに属する
全データ数（questions）を併記して出力します。
"""

import json
import re
from collections import defaultdict

# --- 対応表の定義 ---
TASK_CONSOLIDATION = {
    "osl": ["osl", "horl", "irl"],
    "atl": ["atl", "tl"],
    "oc":  ["oc", "dotc", "dic"],
    "dhoi": ["dhoi", "dtoi"],
    "onvi": ["onvi", "invi"],
    "ndp": ["ndp", "dp"],
    "ii": ["ii"],
    "ai": ["ai"],
    "si": ["si"],
    "sai": ["sai"],
    "asi": ["asi"],
    "itl": ["itl"],
    "nap": ["nap"],
    "npp": ["npp"],
    "nip": ["nip"]
}

FULL_NAMES = {
    "osl": "Object Spatial Localization (horl, irl)",
    "atl": "Action Temporal Localization (tl)",
    "oc":  "Object Counting (dotc, dic)",
    "dhoi": "Dominant Held-Object Identification (dtoi)",
    "onvi": "Object Not Visible Identification (invi)",
    "ndp": "Next Direction Prediction (dp)",
    "ii": "Interaction Identification",
    "ai": "Animal Identification",
    "si": "Sport Identification",
    "sai": "Special Action Identification",
    "asi": "Action Sequence Identification",
    "itl": "Interaction Temporal Localization",
    "nap": "Next Action Prediction",
    "npp": "Next Phase Prediction",
    "nip": "Next Interaction Prediction"
}

def clean_question_text(text):
    if not text: return ""
    text = text.strip()
    cleaned = re.split(r'\n?\s*[A-D][\.\)\:\s]', text)[0]
    return cleaned.strip()

def get_domain_from_item(item):
    if "domain" in item: return item["domain"]
    paths = item.get("video_path") or item.get("images")
    if not paths: return "unknown"
    p = paths[0]
    mapping = {"EgoPet": "animal", "ENIGMA": "industry", "Extrame": "xsports"}
    for k, v in mapping.items():
        if k in p: return v
    return "surgery" if any(x in p for x in ["EgoSurgery", "Cholec"]) else "unknown"

def get_subtask_group(item):
    paths = item.get("video_path") or item.get("images")
    if not paths: return "unknown"
    path_str = paths[0]
    match = re.search(r'/([a-z]+)_q\d+/', path_str)
    if not match: return "unknown"
    raw_key = match.group(1).lower()
    for test_key, support_keys in TASK_CONSOLIDATION.items():
        if raw_key == test_key or raw_key in support_keys:
            return test_key
    return raw_key

def analyze_with_counts(support_path, testbed_path):
    try:
        with open(support_path, "r", encoding="utf-8") as f:
            support_data = json.load(f)
        with open(testbed_path, "r", encoding="utf-8") as f:
            testbed_data = json.load(f)
    except Exception as e:
        print(f"Error: {e}"); return

    # [Domain][GroupKey][ds_type] = { "core_patterns": set(), "total_count": 0 }
    analysis = defaultdict(lambda: defaultdict(lambda: {
        "Support": {"patterns": set(), "total": 0},
        "Testbed": {"patterns": set(), "total": 0}
    }))

    for ds_name, data in [("Support", support_data), ("Testbed", testbed_data)]:
        for item in data:
            dom = get_domain_from_item(item)
            group_key = get_subtask_group(item)
            core_q = clean_question_text(item.get("question_text") or item.get("question", ""))
            
            if core_q:
                analysis[dom][group_key][ds_name]["patterns"].add(core_q)
                analysis[dom][group_key][ds_name]["total"] += 1

    # 出力
    print("="*100)
    print("EgoCross Analysis: Unique Patterns (types) vs Total Data (questions)")
    print("="*100)

    for dom in sorted(analysis.keys()):
        print(f"\n■ DOMAIN: {dom.upper()}")
        for group_key in sorted(analysis[dom].keys()):
            data = analysis[dom][group_key]
            full_name = FULL_NAMES.get(group_key, group_key.upper())
            
            print(f"\n  ● Sub-task Group: {full_name}")
            
            for ds_type in ["Support", "Testbed"]:
                patterns = sorted(list(data[ds_type]["patterns"]))
                total_num = data[ds_type]["total"]
                type_num = len(patterns)
                
                if total_num > 0:
                    # ご要望の形式: [DS Patterns: X types / Y questions]
                    print(f"    [{ds_type} Patterns: {type_num} types / {total_num} questions]")
                    for i, q in enumerate(patterns, 1):
                        print(f"      {i}. {q}")
                else:
                    print(f"    [{ds_type} Patterns: 0 types / 0 questions]")
            print("-" * 60)

if __name__ == "__main__":
    SUPPORT_JSON = "support_db.json"
    TESTBED_JSON = "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/egocross_testbed_imgs.json"
    analyze_with_counts(SUPPORT_JSON, TESTBED_JSON)