"""
support_db.json 自体を評価対象とし、modelscopeのQwen3VLクラスを用いて
ドメイン別・サブタスク別精度を集計するスクリプト。
"""

import argparse
import json
import re
import torch
import os
import gc
import math
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
# modelscopeのクラスを使用
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor  # pylint: disable=no-name-in-module

# =====================
# 0. 引数設定
# =====================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on Support Set using ModelScope")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="モデルのパス")
    parser.add_argument("--min_pixels", type=int, default=3136, help="最小ピクセル数")
    parser.add_argument("--max_pixels", type=int, default=50176, help="最大総画素数")
    parser.add_argument("--gpu_id", type=str, default="3", help="使用するGPUのID")
    return parser.parse_args()


ARGS = parse_args()

# [環境設定]
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# サブタスク統合マップ
TASK_CONSOLIDATION = {
    "osl": ["osl", "horl", "irl"], "atl": ["atl", "tl"], "oc": ["oc", "dotc", "dic"],
    "dhoi": ["dhoi", "dtoi"], "onvi": ["onvi", "invi"], "ndp": ["ndp", "dp"],
    "ii": ["ii"], "ai": ["ai"], "si": ["si"], "sai": ["sai"], "asi": ["asi"],
    "itl": ["itl"], "nap": ["nap"], "npp": ["npp"], "nip": ["nip"]
}


def get_subtask_from_item(item):
    paths = item.get("images", [])
    if not paths:
        return "unknown"
    path_str = paths[0]
    match = re.search(r'/([a-z]+)_q\d+/', path_str)
    raw_key = match.group(1).lower() if match else "unknown"
    for test_key, support_keys in TASK_CONSOLIDATION.items():
        if raw_key == test_key or raw_key in support_keys:
            return test_key
    return raw_key


def normalize_answer(text: str) -> str:
    m = re.search(r"[A-D]", text.upper())
    return m.group(0) if m else ""

# [モデルロード] modelscopeライブラリを使用


print(f"Loading model from {ARGS.model_path} with ModelScope...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    ARGS.model_path,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # 推論スクリプトに合わせて追加
    device_map="auto",
    trust_remote_code=True
).eval()
processor = AutoProcessor.from_pretrained(ARGS.model_path, trust_remote_code=True)
main_device = model.device


def infer_support_item(item, target_res):
    """Zero-shot推論実行"""
    q_text = f"{item['question']}\nAnswer with only the letter (A, B, C, or D)."

    imgs = []
    content = []
    for f_path in item["images"]:
        try:
            with Image.open(f_path) as img:
                # 推論スクリプトに合わせリサイズ処理を追加
                imgs.append(img.convert("RGB").resize((target_res, target_res)))
                content.append({"type": "image"})
        except IOError:
            continue

    content.append({"type": "text", "text": q_text})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[imgs] if imgs else None,
        return_tensors="pt",
        min_pixels=ARGS.min_pixels,
        max_pixels=ARGS.max_pixels
    ).to(main_device)

    with torch.no_grad():
        # do_sample=False で決定論的に出力
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False, use_cache=True,
            temperature=None,
            top_p=None,
            top_k=None
            )
        # 入力トークンを除いた部分をデコード
        decoded = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

    del inputs, out
    return normalize_answer(decoded)


def main():
    res_val = int(math.sqrt(ARGS.max_pixels))
    support_path = "support_db.json"

    with open(support_path, "r", encoding="utf-8") as f:
        support_data = json.load(f)

    # 統計用
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    print(f"Starting Support Set Evaluation (Target Resolution: {res_val}x{res_val})")

    for item in tqdm(support_data):
        domain = item.get("domain", "unknown")
        subtask = get_subtask_from_item(item)
        ground_truth = item.get("answer", "").strip()

        try:
            pred = infer_support_item(item, res_val)

            is_correct = pred == ground_truth
            stats[domain][subtask]["total"] += 1
            if is_correct:
                stats[domain][subtask]["correct"] += 1

            # 定期的なGC
            if stats[domain][subtask]["total"] % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error during inference: {e}")
            continue

    # --- 結果表示 ---
    print("\n" + "="*85)
    print(f"{'DOMAIN > SUBTASK':<55} | {'ACCURACY':<12} | {'SAMPLES'}")
    print("="*85)

    overall_correct = 0
    overall_total = 0

    for dom in sorted(stats.keys()):
        dom_total = sum(s["total"] for s in stats[dom].values())
        dom_corr = sum(s["correct"] for s in stats[dom].values())
        dom_acc = dom_corr / dom_total if dom_total > 0 else 0

        print(f"\n● {dom.upper():<53} | {dom_acc:>10.2%} | {dom_total}")
        print("-" * 85)

        for sub in sorted(stats[dom].keys()):
            s_data = stats[dom][sub]
            s_acc = s_data["correct"] / s_data["total"] if s_data["total"] > 0 else 0
            print(f"  - {sub:<51} | {s_acc:>10.2%} | {s_data['total']}")

        overall_correct += dom_corr
        overall_total += dom_total

    print("\n" + "="*85)
    final_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"{'OVERALL PERFORMANCE ON SUPPORT SET':<55} | {final_acc:>10.2%} | {overall_total}")
    print("="*85)

    # 保存
    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "overall_accuracy": final_acc,
        "details": {d: dict(s) for d, s in stats.items()}
    }
    with open("support_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
