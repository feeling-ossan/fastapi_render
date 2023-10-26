from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 学習済みモデルの読み込み
model = pickle.load(open('model_iris', 'rb'))

# トップページ
@app.get('/')
async def index():          # 非同期処理で複数アクセスがあった場合の処理に対応させる
    return {"Iris": 'iris_prediction'}

# POST が送信された時 (入力) と予測値 (出力) の定義
@app.post('/make_predictions')
async def make_predictions(features: iris):
    return({'prediction':str(model.predict([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])[0])})

# API の中身を記述できたら、ターミナルで uvicorn を起動しましょう。
# uvicorn とは、Python の Web サーバと Web アプリケーションを通信させるインターフェースのことです。