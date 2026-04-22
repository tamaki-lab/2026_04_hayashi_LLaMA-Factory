import json
import os


def build_support_db_from_train():
    """
    学習用(train.json)のメッセージ形式を、検索専用のフラットな構造(support_db.json)に変換する。
    検索のノイズになる <image> タグを事前に除去するのが主な目的。
    """
    # 1. 入出力パスの設定
    # 学習に使用したオリジナルのデータセットを指定
    input_path = "/mnt/HDD18TB/hayashi/2026_04_hayashi_LLaMA-Factory/data/egocross/train.json"
    output_filename = "support_db.json"

    # ファイルの存在確認
    if not os.path.exists(input_path):
        print(f"Error: {input_path} が見つかりません。")
        return

    print(f"Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_db = []

    # 2. データの構造変換ループ
    for item in data:
        # LLM学習用（messagesリスト形式）から必要な情報を抽出
        # messages[0] はユーザーの質問、messages[1] はAIの回答
        user_message = item["messages"][0]["content"]
        assistant_message = item["messages"][1]["content"]

        # 3. テキストのクレンジング
        # TF-IDF検索時に "<image>" という文字列自体が一致判定に影響しないよう削除
        clean_question = user_message.replace("<image>", "").strip()
        clean_answer = assistant_message.strip()

        # 4. 検索エンジン用のシンプル構造に再定義
        # Retrieverクラスが直接参照するフィールド（domain, question, answer）を整理
        entry = {
            "id": item.get("id", "unknown"),
            "domain": item.get("domain", "unknown"),
            "question": clean_question,
            "answer": clean_answer,
            "images": item.get("images", [])  # 画像のパス（ファイル名）も保持しておく
        }
        processed_db.append(entry)

    # 5. 整形したデータベースを保存
    # ensure_ascii=False にすることで日本語や特殊文字もそのまま書き出し
    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(processed_db, f_out, indent=2, ensure_ascii=False)

    print(f"成功！ {len(processed_db)} 件のデータを処理しました。")
    print(f"保存先: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    build_support_db_from_train()
