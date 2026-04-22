import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SupportRetriever:
    """
    質問文の類似度に基づいて、サポートセット（お手本データ）から最適な事例を検索するクラス。
    RAG（検索拡張生成）の「検索」部分を担う。
    """
    def __init__(self, db_path="support_db.json"):
        # 1. データベースの読み込み
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"{db_path} が見つかりません。")

        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

        # 検索対象となるテキスト（訓練データ）のリストを作成
        self.all_questions = [item["question"] for item in self.db]

        # 2. TF-IDF ベクトル化の準備
        # ngram_range=(1, 2) により、単語単体（1-gram）と、2単語の連続（2-gram）の両方を考慮。
        # 例：「video game」を「video」と「game」だけでなく、繋がった意味としても捉える。
        # TF-IDFによって辞書を作成している．
        # この後，この辞書を基にテストセットの質問と，サポートセットの全質問をコサイン類似度を用いて比較する
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

        # データベース内の全質問をベクトル（数値の配列）に変換し、検索のベースを作成
        self.db_vectors = self.vectorizer.fit_transform(self.all_questions)

    def find_top_k(self, query_text, domain=None, k=1):
        """
        クエリ（現在の質問）に対し、類似度が高い上位k件を返す。
        """
        # 3. 入力された質問をベクトルに変換
        query_vector = self.vectorizer.transform([query_text])

        # 4. コサイン類似度の計算
        # ベクトル同士の「向きの近さ」を0.0〜1.0（完全に同じなら1.0）で算出
        similarities = cosine_similarity(query_vector, self.db_vectors).flatten()

        # 5. ドメインによる絞り込み（ドメインフィルタリング）
        # 特定の分野（例：手術）の問題に対し、別の分野（例：動物）の例が出ないようにガードする
        if domain:
            # 内包表記で現在のアイテムが指定ドメインに一致するか判定し、マスクを作成
            mask = np.array([item['domain'] == domain for item in self.db])
            # 一致しない（False）データのスコアを強制的に -1.0 にし、検索対象から外す
            similarities[~mask] = -1.0

        # 6. スコア順にソートして上位インデックスを抽出
        # argsort()[::-1] で、値が大きい順のインデックス配列を得る
        top_indices = similarities.argsort()[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= k:
                break

            # 類似度が負（ドメイン外）のデータまで来たら、そこで取得終了
            if similarities[idx] < 0:
                continue

            # 元のデータと、その類似度スコアをセットにして結果リストに追加
            results.append({
                "item": self.db[idx],
                "score": float(similarities[idx])
            })

        return results


# --- 使用イメージの解説 ---
if __name__ == "__main__":
    # エンジンの起動（DBの全データをベクトル空間に配置）
    retriever = SupportRetriever("support_db.json")

    # 例：現在のテスト問題
    test_q = "What type of animal is featured in this egocentric video segment?"
    test_domain = "animal"

    # 検索の実行
    hits = retriever.find_top_k(test_q, domain=test_domain, k=1)

    if hits:
        # 最も類似度の高いデータを「ヒント」として採用
        best_hit = hits[0]["item"]
        print(f"--- Found Support Item (Score: {hits[0]['score']:.4f}) ---")
        print(f"Domain  : {best_hit['domain']}")
        print(f"Question: {best_hit['question']}")
        print(f"Answer  : {best_hit['answer']}")
    else:
        print("一致するドメインのデータが見つかりませんでした。")
