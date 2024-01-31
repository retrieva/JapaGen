# 日本語タスクにおけるLLMを用いた疑似学習データ生成の検討
NLP2024で発表する「日本語タスクにおけるLLMを用いた疑似学習データ生成の検討」で用いたプロンプトを格納しているリポジトリです。

## 1. 前提
本研究では、LLMにOpneAIのGPT-3.5-TurboのAPIを使用するため、プロンプトは`openai.ChatCompletion.create()`の引数`messages`に入力する以下の形式を想定している。
```python
prompt = [
     {"role": "system", "content": "{タスクの説明}"},
     {"role": "user", "content": "{疑似データ生成のためのインストラクション}"}
]
```

## 2. プロンプト
### 2.1. プロンプト作成のためのConfigファイル
各タスクのConfigファイルをもとにプロンプトを作成する。全タスクを通してConfigファイルの構造は以下のようになっている。
```python
task_conf = {
    "task_name": "{タスク名}",
    "content": "{タスクの説明や設定など}",
    "first instruction": "{1文目生成のためのインストラクション(2文入力タスクの場合)}",
    "labels": {
      "0": {
        "instruction": "{ラベル0に対応する文を生成するためのインストラクション}:"
      },
      "1": {
        "instruction": "{ラベル1に対応する文を生成するためのインストラクション}:"
      }
    }
  }
```
上記のConfigファイルから以下のようにプロンプトを作成し、疑似データを生成する。
```python
label = "0"

# 1文入力タスクの場合
prompt = [
     {"role": "system", "content": task_conf['content']},
     {"role": "user", "content": task_conf['labels'][label]['instruction']}
]
response = openai.ChatCompletion.create(messages=prompt)
generated_text = response['choices'][0]['message']['content']

# 2文入力タスクの場合
prompt1 = [
     {"role": "system", "content": task_conf['content']},
     {"role": "user", "content": task_conf['first instruction']}
]
response1 = openai.ChatCompletion.create(messages=prompt1)
generated_text1 = response1['choices'][0]['message']['content']
prompt2 = [
     {"role": "system", "content": task_conf['content']},
     {"role": "user", "content": task_conf['first instruction']},
     {"role": "assistant", "content": generated_text1},
     {"role": "user", "content": f"{task_conf['labels'][label]['instruction']}"},
]
response2 = openai.ChatCompletion.create(messages=prompt2)
generated_text2 = response2['choices'][0]['message']['content']
```

### 2.2. 各タスクのConfig
#### MARC-ja
```json
{
    "task_name": "marcja",
    "content": "あなたは通販で購入した商品のレビューを記述しています。\n概要:\n- 商品の5段階評価のうち、1と2をネガティブ、4と5をポジティブとする。\n- 5段階評価は含まず、レビュー文のみで回答してください。\n- 記述は200字以内で行ってください。",
    "labels": {
      "0": {
        "instruction": "商品に対するポジティブなレビュー:"
      },
      "1": {
        "instruction": "商品に対するネガティブなレビュー:"
      }
    }
  }
```
#### JSTS
```json
{
    "task_name": "jsts",
    "content": "あなたは画像の説明文を記述しています。\n概要:\n- 記述はそれぞれ50字以内で行ってください。\n- 記述は1文で行ってください。",
    "first instruction": "見ている画像の説明文:",
    "labels": {
      "0": {
        "instruction": "その文と全く関係のない文:"
      },
      "1": {
        "instruction": "その文に出現する単語は含まれるが、関係のない文:"
      },
      "2": {
        "instruction": "その文と少し関係のある文:"
      },
      "3": {
        "instruction": "その文と関係のある文:"
      },
      "4": {
        "instruction": "その文と類似する文:"
      },
      "5": {
        "instruction": "その文と全く内容が同じ文:"
      }
    }
  }
```
#### JNLI
```json
{
    "task_name": "jnli",
    "content": "あなたは画像の説明文を記述しています。\n概要:\n- 記述はそれぞれ50字以内で行ってください。\n- 記述は1文で行ってください。",
    "first instruction": "見ている画像の説明文:",
    "labels": {
      "0": {
        "instruction": "その文に対して含意関係にある説明文:"
      },
      "1": {
        "instruction": "その文に対して矛盾関係にある説明文:"
      },
      "2": {
        "instruction": "その文に対して中立関係にある説明文:"
      }
    }
  }
```
#### JCoLA
```json
{
    "task_name": "jcola",
    "content": "あなたは日本語の教師です。\n概要:\n- 日本語の例文を用いて授業をしています。\n- 例文は50字以内で生成してください。",
    "labels": {
        "0": {
            "instruction": "授業に使用するために統語上誤った日本語文を1文生成してください。"
        },
        "1": {
            "instruction": "授業に使用するために統語上正しい日本語文を1文生成してください。"
        }
    }
}
```
#### News
```json
{
    "task_name": "news",
    "content": "あなたはlivedoor Newsのライターです。\n概要:\n- 「」で指定するトピックのニュース記事を日本語で書いてください。\n- 約200字で生成してください。\n- 指定したトピック名は生成文に含めないでください。",
    "labels": {
      "0": {
        "instruction": "「話題になった出来事」に関するニュースの要約文を書いてください。"
      },
      "1": {
        "instruction": "「Sports Watch」に関するニュースの要約文を書いてください。"
      },
      "2": {
        "instruction": "「ITライフハック」に関するニュースの要約文を書いてください。"
      },
      "3": {
        "instruction": "「家電チャンネル」に関するニュースの要約文を書いてください。"
      },
      "4": {
        "instruction": "「MOVIE ENTER」に関するニュースの要約文を書いてください。"
      },
      "5": {
        "instruction": "「独女通信」に関するニュースの要約文を書いてください。"
      },
      "6": {
        "instruction": "「エスマックス」に関するニュースの要約文を書いてください。"
      },
      "7": {
        "instruction": "「livedoor HOMME」に関するニュースの要約文を書いてください。"
      },
      "8": {
        "instruction": "「Peachy」に関するニュースの要約文を書いてください。"
      }
    }
  }
```
#### COVID-19
```json
{
    "task_name": "covid-19",
    "content": "あなたはCOVID-19に関するツイートをしています。\n概要:\n- 100字以内で生成してください。",
    "labels": {
      "0": {
        "instruction": "ニュースなどの一般的に公表されている情報のみを含むツイートをしてください。"
      },
      "1": {
        "instruction": "あなたの周りで生じている個人的な情報を含むツイートをしてください。"
      },
      "2": {
        "instruction": "意見や感想を含むツイートをしてください。"
      },
      "3": {
        "instruction": "COVID-19とは関係のないツイートをしてください。"
      }
    }
  }
```
#### MARC-ja*
```json
{
    "task_name": "marcja",
    "content": "あなたは通販で購入した商品のレビューを記述しています。\n概要:\n- 商品の5段階評価のうち、1と2をネガティブ、4と5をポジティブとする。\n- 5段階評価は含まず、レビュー文のみで回答してください。\n- 具体的な商品を思い浮かべてください。\n- 記述は200字以内で行ってください。",
    "genre": ["本・漫画・雑誌","DVD・ミュージック・ゲーム","家電・カメラ","パソコン・オフィス用品","ホーム・キッチン","食品・飲料","ヘルス・ビューティー","ベビー・おもちゃ・ホビー","ファッション・時計","スポポーツ・アウトドア","DIY・工具・車用品"],
    "labels": {
      "0": {
        "instruction": "商品（カテゴリ：\"genre\"）に対するポジティブなレビュー:"
      },
      "1": {
        "instruction": "商品（カテゴリ：\"genre\"）に対するネガティブなレビュー:"
      }
    }
  }
```
#### JSTS*
```json
{
    "task_name": "jsts",
    "content": "あなたは画像の説明文を記述しています。\n概要:\n- 画像には\"object\"が写っています。\n- 記述はそれぞれ50字以内で行ってください。\n- 記述は1文で行ってください。",
    "object": ["人々","景色","スポーツ用品","食べ物","雑貨","家具","動物","電気機器","自動車"],
    "first instruction": "見ている画像の説明文:",
    "labels": {
      "0": {
        "instruction": "その文と全く関係のない文:"
      },
      "1": {
        "instruction": "その文に出現する単語は含まれるが、関係のない文:"
      },
      "2": {
        "instruction": "その文と少し関係のある文:"
      },
      "3": {
        "instruction": "その文と関係のある文:"
      },
      "4": {
        "instruction": "その文と類似する文:"
      },
      "5": {
        "instruction": "その文と全く内容が同じ文:"
      }
    }
  }
```
#### JNLI*
```json
{
    "task_name": "jnli",
    "content": "あなたは画像の説明文を記述しています。\n概要:\n- 画像には\"object\"が写っています。\n- 記述はそれぞれ50字以内で行ってください。\n- 記述は1文で行ってください。",
    "object": ["人々","景色","スポーツ用品","食べ物","雑貨","家具","動物","電気機器","自動車"],
    "first instruction": "見ている画像の説明文:",
    "labels": {
      "0": {
        "instruction": "その文に対して含意関係にある説明文:"
      },
      "1": {
        "instruction": "その文に対して矛盾関係にある説明文:"
      },
      "2": {
        "instruction": "その文に対して中立関係にある説明文:"
      }
    }
  }
```
