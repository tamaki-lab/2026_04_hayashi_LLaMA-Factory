import base64
from openai import OpenAI


def encode_image(image_path):
    """画像をBase64形式にエンコードする"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# クライアントの設定
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 1. 画像パスのリスト（10枚）
image_paths = [
    "/mnt/HDD18TB/hayashi/2026_04_hayashi_LLaMA-Factory/data/egocross/frames/EgoPet/022/frames/ai_q5/frame_0.jpg",
]

# 2. メッセージコンテンツの構築
content = []

# 画像を追加
for img_path in image_paths:
    try:
        base64_image = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    except FileNotFoundError:
        print(f"警告: ファイルが見つかりません: {img_path}")

# 3. 質問文を追加
question = """What type of animal is featured in this egocentric video segment?
A. Cheetah
B. Cat
C. Alligator
D. Shark
Answer with a single letter (A, B, C, or D)."""

content.append({"type": "text", "text": question})

# 4. 推論の実行
response = client.chat.completions.create(
    model="egocross",
    messages=[{"role": "user", "content": content}],
    max_tokens=16,
    temperature=0,
    # extra_body を追加して、画像1枚あたりの最大ピクセル数を制限する
    extra_body={
        "max_pixels": 40000  # デフォルトより少し下げることでトークン数を節約
    }
)

# 結果の出力
print(f"Model response: {response.choices[0].message.content}")
