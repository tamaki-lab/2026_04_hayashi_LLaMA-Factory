"""
EgoCrossデータセットに対するQwen3-VLモデルを用いた推論スクリプト。
低FPSの画像列をすべて入力し、解像度（ピクセル数）を制御することでVRAM消費を抑えつつ推論を行います。
"""

# 標準ライブラリおよび外部ライブラリのインポート
import argparse     # コマンドライン引数を解析するためのライブラリ
import gc           # ガベージコレクション（メモリ解放）を制御するライブラリ
import json         # JSON形式のデータの読み書きを行うライブラリ
import os           # ファイルパス操作やディレクトリ作成などのOS依存機能
import re           # 正規表現による文字列パターンの抽出・操作
from datetime import datetime  # 実行日時の取得など、時間関連の機能

# 機械学習・モデル関連のライブラリ
import torch        # PyTorch: テンソル演算およびGPU管理の核となるライブラリ
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration    # pylint: disable=no-name-in-module
from PIL import Image  # 画像ファイルの読み込み・リサイズなどの処理
from tqdm import tqdm  # ループの進捗状況をプログレスバーで表示

import math

# 以下、プログラムの各主要構成要素（モジュール）ごとの処理が記述されます。
# 1. argparseによる実行時引数の設定
# 2. Qwen3-VLモデルおよびプロセッサのロード
# 3. データセット（EgoCross等）の読み込みパス設定
# 4. 推論ループ（VRAM消費抑制のための画像リサイズ処理を含む）
# 5. 結果の保存とメモリのクリーンアップ


def parse_args():
    """
    コマンドライン引数を解析します。
    --max_pixels を調整することで、画像枚数が多い場合のVRAM不足（OOM）を回避できます。
    """
    # 引数解析器の初期化: プログラムの説明文を設定
    parser = argparse.ArgumentParser(description="Inference for EgoCross with adjustable pixel limits.")
    # モデルパスの設定: 推論に使用するマージ済みLoRAモデル等の格納場所を指定
    parser.add_argument("--model_path", type=str, default="./output/egocross_lora_merged_final_test",
                        help="学習済みモデルのディレクトリパス")
    # 最小ピクセル数の設定: 画像をリサイズする際の下限値を指定
    parser.add_argument("--min_pixels", type=int, default=3136, help="最小ピクセル数 (デフォルト: 3136)")
    # 最大ピクセル数の設定: VRAM消費量に直結する重要なパラメータ。画像1枚あたりの解像度を制限
    parser.add_argument("--max_pixels", type=int, default=50176, help="最大ピクセル数 (デフォルト: 224x224相当)")
    parser.add_argument("--gpu_id", type=str, default="3", help="使用するGPUのID")
    # コマンドラインから渡された引数を解析し、名前空間オブジェクトとして返す
    return parser.parse_args()


# 引数の取得
ARGS = parse_args()

# =====================
# 1. 環境設定 & モデルロード
# =====================

# 使用するGPUの制御: システム上の特定のGPU（インデックス2）のみを使用するように制限します。
# 複数GPUがある環境で特定のカードを占有したい場合に有効な設定です。
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id

# PyTorchのメモリ管理最適化: 'expandable_segments'を有効にすることで、
# メモリの確保と解放による断片化を防ぎ、VRAM不足（OOM）のリスクを軽減します。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ロード開始のログ出力: 引数で指定されたモデルパスを表示します。
print(f"Loading model from: {ARGS.model_path}")

# Qwen3-VLモデル本体のロード処理
model = Qwen3VLForConditionalGeneration.from_pretrained(
    ARGS.model_path,
    # 計算精度とメモリ効率のバランス: bfloat16を使用し、VRAM消費を抑えつつ高速に推論します。
    dtype=torch.bfloat16,
    # アテンション機構の実装指定: "sdpa" (Scaled Dot-Product Attention) を使い、メモリ消費と速度を最適化します。
    attn_implementation="sdpa",
    # デバイスの自動割り当て: モデルの各レイヤーを最適なデバイス（GPU/CPU）に自動配置します。
    device_map="auto"
)

# モデルを評価モードに設定: 学習時のみ必要なDropoutなどの挙動を無効化し、推論結果を安定させます。
model.eval()

# 入力処理用のプロセッサのロード
# 1.テキストのトークナイズ
# 2.画像のプリプロセッシング
# 3.データのパッケージング
processor = AutoProcessor.from_pretrained(ARGS.model_path)

# メインで使用されているデバイス（通常はGPU）の情報を保持しておきます。
MAIN_DEVICE = model.device


# =====================
# 2. ユーティリティ
# =====================
def normalize_answer(text: str) -> str:
    """
    モデルの生成テキストから正規表現で A, B, C, D のいずれかを抽出します。
    見つからない場合はデフォルトで 'A' を返します。
    """
    # 生成されたテキストを upper() で大文字に統一し、
    # re.search を使って文字列の中から "A", "B", "C", "D" のいずれか一文字を検索します。
    match = re.search(r"[A-D]", text.upper())
    # マッチする文字が見つかった場合はその文字（A-D）を返し、
    # 形式外の回答などで見つからなかった場合は、便宜上 "A" をデフォルト値として返します。
    return match.group(0) if match else "A"


# =====================
# 3. 1問推論（画像削減なし・ピクセル制限版）
# =====================
def infer_one(item: dict) -> str:
    """
    1つの問題データに対して推論を行います。
    画像パスの置換、Processorによる変換、モデルによる生成処理が含まれます。
    """
    # 質問文の組み立て:
    # 問題文、4つの選択肢、そして「1文字（A, B, C, D）で答えて」という指示を統合します。
    question_text = (
        f"{item['question_text']}\n\n"
        f"{'\n'.join(item['options'])}\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )

    raw_paths = item["video_path"]
    content = []        # Processorに渡す構造（画像とテキストの順序）を保持
    images_list = []    # 実際の画像データを保持

    # 画像列の読み込み処理
    for img_path in raw_paths:
        # パスの置換: JSON内の汎用的なパスを、現在の実行サーバー上の物理パスに書き換えます。
        full_path = img_path.replace("/egocross_testbed/", "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/")
        try:
            with Image.open(full_path) as img:
                # RGB変換: 透過情報（Alpha）などを含む場合でも、モデルが扱える3チャンネル形式に統一します。
                images_list.append(img.convert("RGB"))
                # Qwen3-VLの仕様に合わせ、画像の位置にプレースホルダを挿入します。
                content.append({"type": "image"})
        except FileNotFoundError as error:
            print(f"Skipping: {full_path}, {error}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"Error loading image {full_path}: {error}")

    # メッセージ構造（Qwen3-VLの対話フォーマット）を完成
    content.append({"type": "text", "text": question_text})
    messages = [{"role": "user", "content": content}]

    # チャット用テンプレートの適用:
    # モデルが学習時に使用した特別な区切り文字（<|im_start|>など）を付加します。
    # add_generation_prompt=True により<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...の形式を生成
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Processorによるテンソル化とVRAM制御:
    # ARGS.max_pixels を渡すことで、画像1枚あたりの「視覚トークン数」を抑制し、OOMを防ぎます。
    inputs = processor(
        text=[prompt],
        images=[images_list] if images_list else None,
        padding=True,
        return_tensors="pt",
        min_pixels=ARGS.min_pixels,
        max_pixels=ARGS.max_pixels
    ).to(MAIN_DEVICE)

    # 推論実行（メモリ節約のため勾配計算をオフ）
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,  # 回答は記号1文字なので、生成トークン数は最小限に設定
            do_sample=False,    # 決定論的な回答（常に一番確率が高いもの）を選択
            use_cache=True      # 推論を高速化
        )

    # =====================================================
    # デコード処理（数値列をテキストに変換する工程）
    # =====================================================

    # 1. 入力トークン数の取得:
    # モデルに投げた「質問＋画像情報」が何単語（トークン）分あったかを数えます。
    # inputs.input_ids.shape[1] は、入力データの「長さ」を表します。
    input_len = inputs.input_ids.shape[1]

    # 2. 回答部分の切り出しとデコード:
    # model.generate の戻り値（generated_ids）には、「入力された質問」と「生成された回答」が連結されています。
    # そのため、[ : , input_len : ] と指定することで、入力分をスキップし、
    # モデルが新しく生成したトークン（回答部分）のみをスライスして抽出します。
    decoded = processor.batch_decode(
        generated_ids[:, input_len:],        # 入力後の「新しく生成された部分」だけを渡す
        skip_special_tokens=True,            # <|endoftext|> などのシステム用特殊文字を削除
        clean_up_tokenization_spaces=False   # 単語間のスペースを自動調整せず、モデルの出力を忠実に再現
    )[0]   # バッチ処理（複数同時推論）形式で返るため、最初の1件目([0])を取得

    # =====================================================
    # 回答の正規化（後処理）
    # =====================================================

    # 3. 最終的な記号の抽出:
    # モデルが「正解は B です」のように答えた場合でも、
    # normalize_answer 関数（正規表現）を通して "B" という1文字だけを取り出します。
    pred = normalize_answer(decoded)

    # 明示的なメモリ管理:
    # PyTorchのキャッシュに頼らず、使い終わった大きなテンソルや画像オブジェクトを即座に破棄します。
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
    # 1. パス（ファイルの場所）の定義
    # 推論対象となる画像パスや質問が含まれたメインのデータセットJSON
    testbed_path = "/mnt/HDD18TB/hayashi/data/EgoCross/egocross_testbed/egocross_testbed_imgs.json"
    # 提出フォーマットが定義された空（あるいは初期状態）のJSON
    template_path = "/mnt/HDD18TB/hayashi/2026_04_hayashi_LLaMA-Factory/submission_template.json"
    # 実験手法の名前（ファイル名や管理用に使用）
    method_name = "zeroshot"

    # --- 解像度の計算 ---
    # max_pixels から 1辺の解像度を計算（例: 50176 -> 224）
    resolution = int(math.sqrt(ARGS.max_pixels))

    # 2. 出力準備（保存先ディレクトリとファイル名の決定）
    # 結果を保存するためのフォルダパスを作成
    output_dir = os.path.join("submissions", "zeroshot")
    # フォルダが存在しない場合は自動で作成（エラー防止）
    os.makedirs(output_dir, exist_ok=True)
    # 実行時の日時を取得（例: 20260422_1700）。上書きを防ぎ、実験履歴を管理しやすくします。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # 最終的な保存先のフルパス（例: submissions/zeroshot/20260422_1700_zeroshot.json）
    output_path = os.path.join(output_dir, f"{timestamp}_{method_name}_r{resolution}.json")
    # 3. JSONデータのロード
    # テストデータを読み込み、Pythonのリスト/辞書形式に変換
    with open(testbed_path, "r", encoding="utf-8") as file:
        testbed = json.load(file)
    # 提出用テンプレートを読み込み
    with open(template_path, "r", encoding="utf-8") as file:
        submission = json.load(file)

    # 4. 検索の高速化（ハッシュマップ化）
    # 提出用リストを ID をキーとした辞書に変換します。
    # これにより、特定の ID のデータを探す際、リストを端から走査(O(N))する必要がなくなり、
    # 一瞬(O(1))で見つけることができるようになります（大量データ時に非常に重要）。
    submission_map = {item["id"]: item for item in submission}

    # 5. 保存間隔の設定
    # 全ての処理が終わるのを待たずに、10問ごとに結果をファイルに書き出します。
    # これにより、途中でプログラムが落ちても、それまでの進捗が失われません。
    save_interval = 10

    print(f"Starting inference with max_pixels={ARGS.max_pixels}...")

    # tqdmによる進捗表示:
    # testbed（全問題データ）を1つずつ取り出し、同時にi（0から始まるインデックス）を付与します。
    # tqdmはコンソール上に「あと何分で終わるか」などの進捗状況を表示します。
    for i, item in enumerate(tqdm(testbed, desc="Inference")):
        try:
            # 1. IDの取得とバリデーション
            qid = item["id"]
            # 提出用のsubmission_mapに対象IDが含まれていない（評価対象外）場合は、処理を飛ばします。
            if qid not in submission_map:
                continue

            # 2. 推論実行
            # 先述のinfer_one関数を呼び出し、モデルから予測結果（A, B, C, Dのいずれか）を取得します。
            pred = infer_one(item)

            # 3. 結果の格納
            # submission_map[qid]は元のsubmissionリスト内にある特定の辞書を指しているため、
            # ここで値を更新すると、最終的な保存対象であるsubmissionの内容も同時に書き換わります。
            submission_map[qid]["answer"] = pred

            # 4. 指定間隔での定期保存とキャッシュクリア
            # (i + 1) % save_interval == 0 は「10件ごと」などのタイミングを判定します。
            if (i + 1) % save_interval == 0:
                # 途中結果をファイルへ書き出し。
                # 万が一、この後にサーバーが停電などで落ちても、ここまでのデータは保護されます。
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(submission, file, indent=2)

                # --- VRAM管理の重要ステップ ---
                # torch.cuda.empty_cache(): PyTorchが「念のため確保しているが今は使っていない」ビデオメモリをOSに返却します。
                # gc.collect(): Pythonのゴミ箱（不要になった変数）を掃除し、メインメモリ（RAM）を空けます。
                # これにより、画像列の処理で蓄積しがちなメモリ負荷をリセットします。
                torch.cuda.empty_cache()
                gc.collect()

        # 5. メモリ不足（OOM: Out Of Memory）への対策
        except torch.cuda.OutOfMemoryError:
            # 推論中にVRAMが限界を超えた場合の処理。
            # プログラム全体を停止（クラッシュ）させず、警告を出して次の問題へスキップします。
            print(f"\n⚠️ OOM at ID {item['id']}. Try reducing --max_pixels.")
            torch.cuda.empty_cache()  # 溜まったメモリを強制開放
            gc.collect()
            continue  # 次の問題へ進む

        # 6. その他の予期せぬエラーへの対策
        except Exception as error:  # pylint: disable=broad-except
            # 画像ファイルが壊れている、ネットワークが途切れるなどの想定外の事態。
            # エラー内容を表示して、同様に次の問題へ処理を継続します。
            print(f"\n❌ Error at ID {item['id']}: {error}")
            continue

    # 最終結果の保存
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(submission, file, indent=2)

    print(f"\n✅ 完了！結果保存先: {output_path}")


if __name__ == "__main__":
    main()
