"""
EgoCrossデータセットに対するQwen3-VLモデルを用いた推論スクリプト。
低FPSの画像列をすべて入力し、解像度（ピクセル数）を制御することでVRAM消費を抑えつつ推論を行います。
"""

import argparse
import gc
import json
import os
import re
from datetime import datetime

import torch
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration    # pylint: disable=no-name-in-module
from PIL import Image
from tqdm import tqdm


def parse_args():
    """
    コマンドライン引数を解析します。
    --max_pixels を調整することで、画像枚数が多い場合のVRAM不足（OOM）を回避できます。
    """
    parser = argparse.ArgumentParser(description="Inference for EgoCross with adjustable pixel limits.")
    parser.add_argument("--min_pixels", type=int, default=3136, help="最小ピクセル数 (デフォルト: 3136)")
    parser.add_argument("--max_pixels", type=int, default=50176, help="最大ピクセル数 (デフォルト: 224x224相当)")
    parser.add_argument("--model_path", type=str, default="./output/egocross_lora_merged_final_test",
                        help="学習済みモデルのディレクトリパス")
    return parser.parse_args()


# 引数の取得
ARGS = parse_args()

# =====================
# 1. 環境設定 & モデルロード
# =====================
# GPUの指定（"2"番のみを可視化。外部から指定する場合はここをコメントアウトしてください）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# メモリ断片化を抑制する設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"Loading model from: {ARGS.model_path}")
# Qwen3-VLのロード。dtypeにtorch.bfloat16を指定し、推論の精度と速度を両立。
model = Qwen3VLForConditionalGeneration.from_pretrained(
    ARGS.model_path,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)
model.eval()

# 画像とテキストの処理を司るProcessorのロード
processor = AutoProcessor.from_pretrained(ARGS.model_path)
MAIN_DEVICE = model.device


# =====================
# 2. ユーティリティ
# =====================
def normalize_answer(text: str) -> str:
    """
    モデルの生成テキストから正規表現で A, B, C, D のいずれかを抽出します。
    見つからない場合はデフォルトで 'A' を返します。
    """
    match = re.search(r"[A-D]", text.upper())
    return match.group(0) if match else "A"


# =====================
# 3. 1問推論（画像削減なし・ピクセル制限版）
# =====================
def infer_one(item: dict) -> str:
    """
    1つの問題データに対して推論を行います。
    画像パスの置換、Processorによる変換、モデルによる生成処理が含まれます。
    """
    # 質問文の組み立て（問題文 + 選択肢 + 回答形式の指示）
    question_text = (
        f"{item['question_text']}\n\n"
        f"{'\n'.join(item['options'])}\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )

    raw_paths = item["video_path"]
    content = []
    images_list = []

    # 画像列の読み込み（低FPSを想定し、間引きなしですべて読み込む）
    for img_path in raw_paths:
        # パスの置換（JSON内の相対パスをサーバー上の絶対パスへ変換）
        full_path = img_path.replace("/egocross_testbed/", "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/")
        try:
            with Image.open(full_path) as img:
                # RGB変換。Qwen3-VLは動的解像度に対応しているため、Processor側でリサイズを制御
                images_list.append(img.convert("RGB"))
                content.append({"type": "image"})
        except FileNotFoundError as error:
            print(f"Skipping: {full_path}, {error}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"Error loading image {full_path}: {error}")

    # メッセージ構造（Qwen3-VLのフォーマット）を作成
    content.append({"type": "text", "text": question_text})
    messages = [{"role": "user", "content": content}]

    # Processorを用いてモデル入力形式（テンソル）に変換
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[images_list] if images_list else None,
        padding=True,
        return_tensors="pt",
        min_pixels=ARGS.min_pixels,
        max_pixels=ARGS.max_pixels
    ).to(MAIN_DEVICE)

    # 推論実行（勾配計算を無効化）
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True
        )

        # 入力トークンを除外して生成された回答部分のみをデコード
        input_len = inputs.input_ids.shape[1]
        decoded = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    # 回答の整形
    pred = normalize_answer(decoded)

    # 明示的なメモリ解放（大量の画像を扱うため重要）
    del inputs, generated_ids
    for img in images_list:
        img.close()
    del images_list

    return pred


# =====================
# 4. メイン処理
# =====================
def main():
    """
    メインループ。テスト用JSONを読み込み、定期的に進捗を保存しながら推論を進めます。
    """
    testbed_path = "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/egocross_testbed_imgs.json"
    template_path = "/mnt/HDD18TB/hayashi/2026_04_hayashi_LLaMA-Factory/submission_template.json"
    method_name = "zeroshot"

    # 出力ディレクトリ作成
    output_dir = os.path.join("submissions", "zeroshot")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(output_dir, f"{timestamp}_{method_name}.json")

    # JSONデータのロード
    with open(testbed_path, "r", encoding="utf-8") as file:
        testbed = json.load(file)
    with open(template_path, "r", encoding="utf-8") as file:
        submission = json.load(file)

    # 提出用データのマップ化（ID検索の高速化）
    submission_map = {item["id"]: item for item in submission}
    save_interval = 10

    print(f"Starting inference with max_pixels={ARGS.max_pixels}...")

    # tqdmによる進捗表示
    for i, item in enumerate(tqdm(testbed, desc="Inference")):
        try:
            qid = item["id"]
            if qid not in submission_map:
                continue

            # 推論実行
            pred = infer_one(item)
            submission_map[qid]["answer"] = pred

            # 指定間隔での定期保存とキャッシュクリア
            if (i + 1) % save_interval == 0:
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(submission, file, indent=2)
                torch.cuda.empty_cache()
                gc.collect()

        except torch.cuda.OutOfMemoryError:
            print(f"\n⚠️ OOM at ID {item['id']}. Try reducing --max_pixels.")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as error:  # pylint: disable=broad-except
            print(f"\n❌ Error at ID {item['id']}: {error}")
            continue

    # 最終結果の保存
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(submission, file, indent=2)

    print(f"\n✅ 完了！結果保存先: {output_path}")


if __name__ == "__main__":
    main()
