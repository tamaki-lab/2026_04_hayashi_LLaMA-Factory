"""
EgoCrossデータセットに対するQwen3-VLモデルを用いた Dynamic Few-shot 推論スクリプト。
argparseにより解像度、GPU、お手本数、モデルパスを柔軟に設定可能です。
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
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor  # pylint: disable=no-name-in-module
from retriever import SupportRetriever

# =====================
# 0. 引数設定
# =====================


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic ICL Inference for EgoCross")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="モデルのパス")
    parser.add_argument("--min_pixels", type=int, default=3136, help="最小ピクセル数 (デフォルト: 3136)")
    parser.add_argument("--max_pixels", type=int, default=50176, help="最大総画素数 (例: 50176 は 224x224)")
    parser.add_argument("--num_shots", type=int, default=2, help="検索するお手本の数")
    parser.add_argument("--gpu_id", type=str, default="3", help="使用するGPUのID")
    return parser.parse_args()


ARGS = parse_args()

# [環境設定] GPUの指定と、セグメント拡張によるメモリ断片化の抑制
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def normalize_answer(text: str) -> str:
    """正規表現で出力からA-Dの回答記号のみを抽出する"""
    m = re.search(r"[A-D]", text.upper())
    return m.group(0) if m else "A"


def get_domain_from_path(path_list: list) -> str:
    """
    ファイルパスに含まれる文字列からデータ種別（ドメイン）を判定する。
    画像や動画がどのカテゴリ（動物、工業、手術など）に属するかを識別し、
    その後のRetriever（類似例検索）でドメインを限定するために使用される。
    """
    # 1. パスリストが空（画像がない等）の場合は、判定不能として "unknown" を返す
    if not path_list:
        return "unknown"

    # 2. リストの先頭のパスを判定対象として取得（通常、同一アイテム内のパスは同じドメインに属するため）
    p = path_list[0]

    # 3. キーワードとドメイン名のマッピング定義
    # 左側の文字列がパスに含まれていれば、右側のドメイン名（animal等）を割り当てる
    mapping = {"EgoPet": "animal", "ENIGMA": "industry", "Extrame": "xsports"}

    # 4. マッピングに基づきループで判定を実行
    for k, v in mapping.items():
        if k in p:
            return v

    # 5. 特殊な複数キーワード（外科手術関連）の判定
    # "EgoSurgery" または "Cholec"（胆嚢摘出術関連データ）が含まれる場合は "surgery" と判定
    # いずれにも該当しない場合は最終的に "unknown" を返す
    return "surgery" if any(x in p for x in ["EgoSurgery", "Cholec"]) else "unknown"


def build_few_shot_text(support_items: list) -> str:
    """
    検索された類似例（サポートセット）を整形して、プロンプトに挿入する関数。
    【目的】
    VLM（視覚言語モデル）に対して「過去の似た問題と正解」をお手本として提示し、
    回答の形式（記号で答える等）や推論の方向性を誘導する（Few-shot prompting）。
    """

    # 事例が見つからなかった（DBが空、あるいは検索失敗など）場合は、
    # お手本なしのゼロショット（通常通り）で解かせるため、空文字を返す。
    if not support_items:
        return ""

    # 1. プロンプトのヘッダー部分を定義
    # モデルに対して「これから例を出すから、それを参考に画像を分析してね」という役割を明示する。
    res = "### Instructions:\nAnalyze frames with reference to examples:\n\n"

    # 2. 検索された類似例（k個）を1つずつ取り出して整形
    for i, s in enumerate(support_items):
        # s['question'] : 過去の質問文（選択肢を含む）
        # s['answer']   : その質問に対する正解ラベル（例: "B"）

        # .strip() を使う理由:
        # DB保存時にも行っているが、ここでも徹底することで余計な改行や空白による
        # プロンプトの崩れを防ぎ、モデルが「A: B」というパターンを認識しやすくする。
        res += f"Ex {i+1}: Q: {s['question'].strip()} A: {s['answer'].strip()}\n\n"

    # 3. 現在の問題（テストセット）を解くためのフッターを追加
    # 「ここまでは過去の例、ここからは今解くべきタスク」という境界をはっきりさせる。
    # この後に「Q: (現在の質問)」が続くことで、モデルはこれまでの例に倣って回答を生成する。
    return res + "### Current Task:\nSolve based on provided images:\n"


# [モデル準備] Qwen3-VLをbf16精度でロード。sdpaで推論を高速化
model = Qwen3VLForConditionalGeneration.from_pretrained(
    ARGS.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
    local_files_only=True
).eval()
processor = AutoProcessor.from_pretrained(ARGS.model_path)
main_device = model.device


def infer_one(item: dict, support_items: list, target_res: int) -> str:
    """画像群とテキストをVLMに入力し、1つの回答を得るメイン推論関数"""
    q_text = (
        f"{build_few_shot_text(support_items)}{item['question_text']}\n"
        f"{''.join(item['options'])}\nAnswer with A, B, C, or D."
    )

    imgs = []
    content = []
    base_dir = "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/"
    for path in item["video_path"]:
        # ローカル環境のフルパスに変換して画像を読み込み・リサイズ
        f_path = path.replace("/egocross_testbed/", base_dir)
        try:
            with Image.open(f_path) as img:
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

    # 1. データのテンソル化（Processorによる前処理）
    # テキスト（prompt）と画像（imgs）を、モデルが計算可能な数値形式（テンソル）に変換します。
    # max_pixels: VLMのメモリ消費を抑えるため、入力画像の最大画素数を制限します。
    inputs = processor(
        text=[prompt],
        images=[imgs] if imgs else None,
        return_tensors="pt",
        min_pixels=ARGS.min_pixels,
        max_pixels=ARGS.max_pixels
    ).to(main_device)

    # 2. モデルによる回答生成（推論実行）
    # torch.no_grad(): 勾配計算を無効化し、メモリ節約と計算の高速化を図ります。
    with torch.no_grad():
        # max_new_tokens=10: 回答は「A」などの1文字で良いため、生成する長さを極短く制限します。
        # do_sample=False: 確率的なゆらぎを排除し、常に最も確率の高い（決定論的な）回答を選びます。
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)

        # 3. 回答のデコード（数値から文字への復元）
        # outには「入力（問題）」＋「出力（回答）」が全て含まれています。
        # inputs.input_ids.shape[1] 以降を指定することで、モデルが新しく生成した「回答部分」のみを抽出します。
        decoded = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

    # 4. 後処理とリソース解放
    # GPUメモリの断片化を防ぐため、使い終わったテンソルを明示的に削除します。
    del inputs, out

    # normalize_answer: 「Result: A」のように余計な文字が入った場合でも、「A」だけを取り出す整形関数を通します。
    return normalize_answer(decoded)


def main():
    """データセットのロード、ドメイン毎の精度集計、定期的保存を実行"""
    # max_pixels から解像度（1辺の長さ）を計算
    res_val = int(math.sqrt(ARGS.max_pixels))

    testbed_p = "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/egocross_testbed_imgs.json"
    retriever = SupportRetriever("support_db.json")

    with open(testbed_p, "r", encoding="utf-8") as f:
        testbed = json.load(f)
    with open("submission_template.json", "r", encoding="utf-8") as f:
        sub = json.load(f)

    sub_map = {item["id"]: item for item in sub}
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    # 実験手法の名前（ファイル名や管理用に使用）
    method_name = "dynamic_icl"

    # 2. 出力準備（保存先ディレクトリとファイル名の決定）
    # 結果を保存するためのフォルダパスを作成
    output_dir = os.path.join("submissions", method_name)
    # フォルダが存在しない場合は自動で作成（エラー防止）
    os.makedirs(output_dir, exist_ok=True)
    # 実行時の日時を取得（例: 20260422_1700）。上書きを防ぎ、実験履歴を管理しやすくします。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # 最終的な保存先のフルパス（例: submissions/dynamic_icl/20260422_1700_dynamic_icl_r224.json）
    output_path = os.path.join(output_dir, f"{timestamp}_{method_name}_r{res_val}.json")

    # testbed（テストデータ全体）を一つずつ取り出してループ
    for i, item in enumerate(tqdm(testbed)):
        try:
            # 提出用マップ（sub_map）に存在しないIDはスキップ
            if item["id"] not in sub_map:
                continue

            # 1. ドメインの特定
            # 動画のパスから、その問題がどのカテゴリ（手術、料理など）に属するかを判定
            dom = get_domain_from_path(item["video_path"])

            # 2. 類似事例の検索（RAG: Retrieval Augmented Generation）
            # Retriever（TF-IDF）を使い、今の質問文に似ている過去の事例を上位2件（k=2）取得
            # domainを指定することで、同じカテゴリ内からのみ検索し、精度の高いお手本を探す
            hits = [h['item'] for h in retriever.find_top_k(item["question_text"], domain=dom, k=ARGS.num_shots)]

            # 3. VLMによる推論
            # 「今の問題(item)」と「お手本(hits)」をプロンプトとしてVLMに入力し、回答(A, B, C, D)を得る
            pred = infer_one(item, hits, res_val)

            # 4. 回答の保存
            # 推論結果を提出用フォーマットのマップに格納
            sub_map[item["id"]]["answer"] = pred

            # 5. 統計の集計（正解がわかっている場合のみ）
            if "answer" in item:
                stats[dom]["total"] += 1
                if pred == item["answer"]:
                    stats[dom]["correct"] += 1

            # 50件ごとにメモリ解放と中間保存を行い、長時間実行の安全性を確保
            if (i + 1) % 50 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sub, f, indent=2)
                gc.collect()
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error at {i}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=2)


if __name__ == "__main__":
    main()
