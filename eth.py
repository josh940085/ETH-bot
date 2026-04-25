# ===== 移除 gevent（穩定版）=====
import requests
import datetime
import time
import pandas as pd
import numpy as np
import threading
import websocket
import json
import pickle
import os
import sys
import re
import html
import subprocess
import hmac
import hashlib
import urllib.parse
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from threading import Lock

# ===== 手續費設定 =====
# Binance 永續合約 Taker 手續費（單邊 0.04%）
# 開倉 + 平倉合計 = 0.08%，TP 必須覆蓋此費用才能獲利
TAKER_FEE_RATE = 0.0004          # 單邊 0.04%
ROUND_TRIP_FEE_RATE = TAKER_FEE_RATE * 2  # 雙邊合計 0.08%

# ===== Macro / News Engine =====

MACRO_CACHE = {"sp": 0, "nq": 0, "btc": 0, "dxy": 0, "news": 0, "event": 0, "news_list": [], "ts": 0}
NEWS_CACHE = {"news": 0, "event": 0, "news_list": [], "ts": 0}

# ===== AI 新聞分類 =====
NEWS_MODEL_PATH = "news_model.pkl"
NEWS_VECTORIZER_PATH = "news_vectorizer.pkl"
NEWS_PERFORMANCE_LOG = "news_predictions.jsonl"   # 記錄所有預測結果用於評估
NEWS_LEARNING_BUFFER = "learning_buffer.pkl"       # 增量學習緩衝區
news_model = None
news_vectorizer = None

# 增量學習配置
INCREMENTAL_LEARNING_ENABLED = True
MIN_PREDICTIONS_FOR_RETRAIN = 50  # 每50個預測後考慮重新訓練

HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

TRANSLATION_CACHE = {}
BOT_SOFT_RESTART_REQUESTED = False
TELEGRAM_STATE_PATH = Path(__file__).resolve().parent / ".telegram_state.json"

# ===== Environment variables / secrets =====
def _load_local_env():
    """簡易讀取 .env（不依賴 python-dotenv）。"""
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as e:
        print(f"⚠️ .env 載入失敗: {e}")


def _get_required_env(name, default=None, mask=False):
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        print(f"⚠️ 缺少環境變數: {name}")
        return default
    if mask:
        print(f"✅ 已載入 {name}")
    return value


_load_local_env()

TELEGRAM_TOKEN = _get_required_env("TELEGRAM_TOKEN", "", mask=True)
TELEGRAM_CHAT_ID = _get_required_env("TELEGRAM_CHAT_ID", "")
# ===== Telegram =====
LAST_TELEGRAM_TS = 0
TELEGRAM_PINNED_MESSAGE_ID = None

# ===== Discord（同步通知） =====
DISCORD_WEBHOOK = _get_required_env("DISCORD_WEBHOOK", "", mask=True)

# ===== Binance Futures 交易 API =====
BINANCE_API_KEY = _get_required_env("BINANCE_API_KEY", "", mask=True)
BINANCE_API_SECRET = _get_required_env("BINANCE_API_SECRET", "", mask=True)
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_PAPI_BASE = "https://papi.binance.com"
# ETHUSDT 最小下單數量（Binance 規定 0.001 ETH）
_MIN_ORDER_QTY = 0.001
# 預設槓桿倍數（可透過環境變數 BINANCE_LEVERAGE 覆蓋）
BINANCE_LEVERAGE = int(os.environ.get("BINANCE_LEVERAGE", "10"))


def _normalize_finance_terms_zh(text):
    """將翻譯結果統一為較常見的股市/總經術語。"""
    s = str(text or "").strip()
    if not s:
        return s

    # 英文財經詞彙（先做，避免混雜）
    en_map = [
        (r"\bWall\s*St\.?\b", "華爾街"),
        (r"\bconsumer prices?\b", "消費者物價"),
        (r"\binflation\b", "通膨"),
        (r"\bcore inflation\b", "核心通膨"),
        (r"\bTreasury yields?\b", "美債殖利率"),
        (r"\bfutures?\b", "期貨"),
        (r"\bstock rating\b", "投資評級"),
        (r"\brating\b", "評級"),
        (r"\bprice target\b", "目標價"),
        (r"\breiterates?\b", "重申"),
        (r"\bcuts?\b", "下調"),
        (r"\braises?\b", "上調"),
        (r"\bmaintains?\b", "維持"),
        (r"\binitiates?\b", "首次覆蓋"),
        (r"\bheadwinds?\b", "逆風"),
        (r"\bupgrades?\b", "上調評級"),
        (r"\bdowngrades?\b", "下調評級"),
        (r"\bearnings\b", "財報"),
        (r"\bguidance\b", "財測"),
        (r"\boutlook\b", "展望"),
        (r"\bestimate\b", "預估"),
        (r"\bestimates\b", "預估"),
        (r"\bforecast\b", "預測"),
        (r"\bforecasts\b", "預測"),
        (r"\brevenue\b", "營收"),
        (r"\bprofit\b", "獲利"),
        (r"\bmargin\b", "利潤率"),
        (r"\bEPS\b", "每股盈餘"),
        (r"\bvaluation\b", "估值"),
        (r"\bshares?\b", "股價"),
    ]

    for pat, rep in en_map:
        s = re.sub(pat, rep, s, flags=re.I)

    # 中文術語正規化（將口語或陸式寫法統一）
    zh_map = [
        ("股价", "股價"),
        ("评级", "評級"),
        ("投資評等", "投資評級"),
        ("評級重申", "重申評級"),
        ("重申中立評級", "重申中立評級"),
        ("重申減持評級", "重申減碼評級"),
        ("重申增持評級", "重申加碼評級"),
        ("重申跑贏大盤評級", "重申優於大盤評級"),
        ("重申買入評級", "重申買進評級"),
        ("重申买入评级", "重申買進評級"),
        ("買入評級", "買進評級"),
        ("买入评级", "買進評級"),
        ("持有評級", "中立評級"),
        ("買入", "買進"),
        ("卖出", "賣出"),
        ("增持", "加碼"),
        ("减持", "減碼"),
        ("下调", "下調"),
        ("上调", "上調"),
        ("目标价", "目標價"),
        ("目标价格", "目標價"),
        ("目標價格", "目標價"),
        ("調升目標價", "上調目標價"),
        ("調降目標價", "下調目標價"),
        ("上修", "上調"),
        ("下修", "下調"),
        ("通货膨胀", "通膨"),
        ("华尔街", "華爾街"),
        ("美国", "美國"),
        ("美联储", "聯準會"),
        ("利率决议", "利率決議"),
        ("非农", "非農"),
        ("收益率", "殖利率"),
        ("業績指引", "財測"),
        ("指引", "財測"),
        ("營業收入", "營收"),
        ("每股收益", "每股盈餘"),
        ("每股盈利", "每股盈餘"),
    ]

    for old, new in zh_map:
        s = s.replace(old, new)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _google_translate_to_zh(text):
    """使用公開 Google 翻譯端點作為第二層備援。"""
    try:
        src = str(text or "").strip()
        if not src:
            return ""

        res = HTTP_SESSION.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "zh-TW",
                "dt": "t",
                "q": src[:350],
            },
            timeout=6,
        )
        data = res.json()
        if not isinstance(data, list) or not data:
            return ""

        parts = data[0] if isinstance(data[0], list) else []
        text_parts = []
        for item in parts:
            if isinstance(item, list) and item:
                seg = str(item[0] or "").strip()
                if seg:
                    text_parts.append(seg)

        out = "".join(text_parts).strip()
        return _normalize_finance_terms_zh(out)
    except Exception:
        return ""


def _local_translate_news_fallback(text):
    """當 API 翻譯失敗時，使用本地詞彙表做可讀的中文轉換。"""
    s = str(text or "").strip()
    if not s:
        return s

    if re.search(r"[\u4e00-\u9fff]", s):
        return s

    table = [
        ("Federal Reserve", "聯準會"),
        ("interest rate", "利率"),
        ("rate cut", "降息"),
        ("rate hike", "升息"),
        ("inflation", "通膨"),
        ("nonfarm payrolls", "非農就業"),
        ("unemployment", "失業率"),
        ("Treasury", "美債"),
        ("yield", "殖利率"),
        ("US Dollar", "美元"),
        ("dollar index", "美元指數"),
        ("Bitcoin", "比特幣"),
        ("Ethereum", "以太幣"),
        ("crypto", "加密貨幣"),
        ("ETF", "ETF"),
        ("approved", "獲批准"),
        ("approval", "批准"),
        ("inflow", "資金流入"),
        ("outflow", "資金流出"),
        ("rally", "上漲"),
        ("surge", "飆升"),
        ("plunge", "大跌"),
        ("drop", "下跌"),
        ("sell-off", "拋售"),
        ("lawsuit", "訴訟"),
        ("hack", "駭客攻擊"),
        ("exchange", "交易所"),
        ("listing", "上架"),
        ("delist", "下架"),
        ("partnership", "合作"),
        ("tariff", "關稅"),
        ("ceasefire", "停火"),
        ("sanction", "制裁"),
    ]

    out = s
    for en, zh in table:
        out = re.sub(re.escape(en), zh, out, flags=re.I)

    if not re.search(r"[\u4e00-\u9fff]", out):
        return f"（原文）{s}"
    return _normalize_finance_terms_zh(out)



def normalize_news_text(text):
    try:
        text = html.unescape(str(text))
        text = text.replace("\\u002F", "/").replace("\\/", "/")
        text = text.replace("\\n", " ").replace("\\t", " ")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except:
        return str(text)

def translate_news_to_zh(text):
    """將新聞標題翻譯成繁中，失敗時回傳原文。"""
    try:
        src = str(text or "").strip()
        if not src:
            return src

        # 已含中文就不重翻
        if re.search(r"[\u4e00-\u9fff]", src):
            return src

        if src in TRANSLATION_CACHE:
            return TRANSLATION_CACHE[src]

        # 保守截斷，避免超長文本增加延遲與成本
        short_src = src[:220]

        zh = ""
        if OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是專業翻譯員。請把輸入內容翻成繁體中文，只輸出翻譯結果，不要補充說明。"
                    },
                    {
                        "role": "user",
                        "content": short_src
                    }
                ],
                "temperature": 0
            }

            res = requests.post(url, headers=headers, json=payload, timeout=6)
            data = res.json() if res is not None else {}
            zh = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not zh:
            zh = _google_translate_to_zh(src)

        if not re.search(r"[\u4e00-\u9fff]", zh):
            zh = _google_translate_to_zh(src)

        if not re.search(r"[\u4e00-\u9fff]", zh):
            zh = _local_translate_news_fallback(src)

        zh = _normalize_finance_terms_zh(zh)

        if len(TRANSLATION_CACHE) > 1000:
            TRANSLATION_CACHE.clear()
        TRANSLATION_CACHE[src] = zh
        return zh
    except Exception:
        zh = _google_translate_to_zh(text)
        if re.search(r"[\u4e00-\u9fff]", zh):
            return zh
        return _local_translate_news_fallback(text)


# ===== AI 新聞分類訓練數據 =====
NEWS_TRAINING_DATA = [
    # 利多 - 強信號
    ("Bitcoin ETF approved by SEC", 2),
    ("Ethereum staking approval approved", 2),
    ("MicroStrategy buys more Bitcoin holdings", 2),
    ("BlackRock files for spot Bitcoin ETF", 2),
    ("Crypto adoption in El Salvador", 2),
    ("Major bank launches crypto trading", 2),
    ("Hash rate reaches all-time high", 2),
    ("Network upgrade goes live successfully", 2),
    ("Institutional investors buy crypto", 2),
    ("New partnership for Bitcoin announced", 2),
    ("Crypto ETF inflow record high", 2),
    ("Spot Bitcoin ETF approval", 2),
    # 利多 - 弱信號
    ("Crypto market rallies higher", 1),
    ("Bitcoin surge", 1),
    ("Ethereum listing on exchange", 1),
    ("Adoption increases", 1),
    ("Record accumulation by whales", 1),
    ("Positive sentiment in market", 1),
    ("Upgrade launches", 1),
    ("Support for crypto regulation", 1),
    # 利空 - 強信號
    ("Crypto exchange hacked stolen funds", -2),
    ("FTX collapse bankruptcy", -2),
    ("SEC charges crypto company fraud", -2),
    ("Major exchange delisted", -2),
    ("Bitcoin hack $100 million stolen", -2),
    ("Lawsuit against crypto firm", -2),
    ("Investigation fraud charges", -2),
    ("Exchange suspends withdrawals", -2),
    ("Regulatory ban announced", -2),
    # 利空 - 弱信號
    ("Crypto market drops lower", -1),
    ("Sell-off in Bitcoin", -1),
    ("Ethereum down", -1),
    ("Outflow from crypto funds", -1),
    ("Bearish sentiment market", -1),
    ("Price decline", -1),
    ("Whale dump", -1),
    # 宏觀 / 事件類
    ("Fed raises interest rates", 0),
    ("FOMC meeting decision", 0),
    ("Inflation data released", 0),
    ("War tensions Middle East", 0),
    ("Tariff announcement", 0),
    ("CPI report misses", 0),
    ("Labor data released", 0),
    ("Economic event scheduled", 0),
    ("Geopolitical news", 0),
    ("Policy announcement", 0),
]

def train_news_model():
    global news_model, news_vectorizer
    if news_model is not None:
        return  # 已訓練

    texts = [item[0] for item in NEWS_TRAINING_DATA]
    labels = [item[1] for item in NEWS_TRAINING_DATA]

    # 改進的特徵提取：使用雙詞組合 + 子線性 TF
    news_vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),      # 捕捉詞組
        min_df=1,
        sublinear_tf=True,        # 改進詞頻計算
        stop_words='english'
    )
    X = news_vectorizer.fit_transform(texts)
    y = np.array(labels)

    # 使用集成投票模型（Gradient Boosting + Random Forest + Logistic Regression）
    news_model = VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
            ('lr', LogisticRegression(max_iter=200, random_state=42))
        ],
        voting='soft'  # 使用概率投票
    )
    news_model.fit(X, y)

    # 保存模型
    try:
        with open(NEWS_MODEL_PATH, "wb") as f:
            pickle.dump(news_model, f)
        with open(NEWS_VECTORIZER_PATH, "wb") as f:
            pickle.dump(news_vectorizer, f)
    except:
        pass

def load_news_model():
    global news_model, news_vectorizer
    try:
        with open(NEWS_MODEL_PATH, "rb") as f:
            news_model = pickle.load(f)
        with open(NEWS_VECTORIZER_PATH, "rb") as f:
            news_vectorizer = pickle.load(f)
    except:
        train_news_model()

def predict_news_sentiment(text):
    """預測新聞情緒（舊函數，保持兼容性）"""
    global news_model, news_vectorizer
    if news_model is None:
        load_news_model()
    if news_model is None:
        return 0  # 預設中性

    X = news_vectorizer.transform([text])
    prediction = news_model.predict(X)[0]
    return int(prediction)


def predict_news_sentiment_with_confidence(text):
    """預測新聞情緒 + 置信度分數（新函數，更智能）"""
    global news_model, news_vectorizer
    if news_model is None:
        load_news_model()
    if news_model is None:
        return 0, 0.33  # 預設中性，低置信度

    try:
        X = news_vectorizer.transform([text])
        prediction = news_model.predict(X)[0]
        
        # 獲取概率分布
        probabilities = news_model.predict_proba(X)[0]
        confidence = max(probabilities)  # 取最高概率
        
        return int(prediction), float(confidence)
    except:
        return 0, 0.33


def _keyword_bias_score(text):
    """中性修正用關鍵字分數：>0 偏多，<0 偏空。"""
    low = str(text or "").lower()

    bull_words = [
        "approval", "approved", "etf", "inflow", "surge", "rally", "breakout",
        "partnership", "adoption", "upgrade", "listing", "launch", "buyback",
        "accumulation", "institutional", "rate cut", "stimulus"
    ]
    bear_words = [
        "hack", "exploit", "lawsuit", "ban", "fraud", "bankruptcy", "delist",
        "outflow", "dump", "sell-off", "crash", "plunge", "investigation",
        "sanction", "rate hike", "liquidation", "withdrawal halt"
    ]

    score = 0
    for w in bull_words:
        if w in low:
            score += 1
    for w in bear_words:
        if w in low:
            score -= 1

    return score


def _refine_neutral_bias(text, ai_bias, ai_confidence):
    """修正中性判斷：AI 低信心或中性時，使用關鍵字進行二次判斷。"""
    final_bias = int(ai_bias)
    k_score = _keyword_bias_score(text)

    # AI 明確高信心時，不強行覆蓋
    if ai_confidence >= 0.68 and abs(final_bias) >= 1:
        return final_bias

    # AI 判中性時，優先用關鍵字修正
    if final_bias == 0:
        if k_score >= 2:
            return 1
        if k_score <= -2:
            return -1
        return 0

    # AI 低信心弱訊號時，關鍵字可升降一級
    if abs(final_bias) == 1 and ai_confidence < 0.52:
        if k_score >= 3:
            return 2
        if k_score <= -3:
            return -2

    return final_bias


# ===== 增量學習系統：記錄預測並持續改進 =====
def log_prediction_result(news_text, predicted_bias, actual_market_move=None, correct=None):
    """記錄預測結果用於增量學習和精準度評估"""
    try:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "news": news_text[:150],
            "predicted_bias": predicted_bias,
            "actual_move": actual_market_move,
            "is_correct": correct
        }
        
        with open(NEWS_PERFORMANCE_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except:
        pass


def get_prediction_accuracy():
    """計算模型預測準確度"""
    try:
        total = 0
        correct = 0
        
        with open(NEWS_PERFORMANCE_LOG, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("is_correct") is not None:
                        total += 1
                        if record["is_correct"]:
                            correct += 1
                except:
                    pass
        
        accuracy = (correct / total * 100) if total > 0 else 0
        return {"accuracy": round(accuracy, 2), "total": total, "correct": correct}
    except:
        return {"accuracy": 0, "total": 0, "correct": 0}


def update_learning_buffer(news_text, true_label):
    """將新樣本添加到增量學習緩衝區"""
    try:
        buffer = []
        try:
            with open(NEWS_LEARNING_BUFFER, "rb") as f:
                buffer = pickle.load(f)
        except:
            buffer = []
        
        buffer.append((news_text, true_label))
        
        # 緩衝區最多保留 200 個樣本
        if len(buffer) > 200:
            buffer = buffer[-200:]
        
        with open(NEWS_LEARNING_BUFFER, "wb") as f:
            pickle.dump(buffer, f)
    except:
        pass


def incremental_train_news_model():
    """增量學習：結合原始訓練數據 + 學習緩衝區新樣本進行重新訓練"""
    global news_model, news_vectorizer
    
    texts = [item[0] for item in NEWS_TRAINING_DATA]
    labels = [item[1] for item in NEWS_TRAINING_DATA]
    
    # 讀取學習緩衝區的新樣本
    try:
        with open(NEWS_LEARNING_BUFFER, "rb") as f:
            buffer = pickle.load(f)
            for text, label in buffer:
                texts.append(text)
                labels.append(label)
    except:
        pass
    
    # 防止訓練數據過多導致過擬合
    if len(texts) > 500:
        texts = texts[-500:]
        labels = labels[-500:]
    
    # 重新訓練模型
    try:
        news_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words='english'
        )
        X = news_vectorizer.fit_transform(texts)
        y = np.array(labels)

        news_model = VotingClassifier(
            estimators=[
                ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
                ('lr', LogisticRegression(max_iter=200, random_state=42))
            ],
            voting='soft'
        )
        news_model.fit(X, y)

        # 保存更新的模型
        with open(NEWS_MODEL_PATH, "wb") as f:
            pickle.dump(news_model, f)
        with open(NEWS_VECTORIZER_PATH, "wb") as f:
            pickle.dump(news_vectorizer, f)
        
        print(f"✓ 增量學習完成：使用 {len(texts)} 個樣本重新訓練模型")
    except Exception as e:
        print(f"✗ 增量學習失敗: {e}")


# 新聞情緒/事件分析（更穩定的分類）
def analyze_news_text(raw_text):
    """更穩定的新聞分類：拆分多空 / 事件 / 影響，避免單一關鍵字誤判。"""
    text = str(raw_text or "").strip()

    # ===== 直接使用 AI 模型判斷，不再依賴關鍵字規則 =====
    ai_bias, ai_confidence = predict_news_sentiment_with_confidence(text)
    tags = [f"ai_conf:{ai_confidence:.2f}"]
    fusion_note = "ai_only"
    final_bias = _refine_neutral_bias(text, ai_bias, ai_confidence)
    event_risk = 0

    if ai_confidence < 0.4:
        tags.append("low_confidence")
    else:
        tags.append("high_confidence")

    if final_bias >= 2:
        sentiment = "偏多 (強)"
        impact = "利多（價格可能上漲）"
    elif final_bias == 1:
        sentiment = "偏多"
        impact = "輕微利多（偏正面）"
    elif final_bias == 0:
        sentiment = "中性"
        impact = "影響有限"
    elif final_bias == -1:
        sentiment = "偏空"
        impact = "輕微利空（偏負面）"
    else:
        sentiment = "偏空 (強)"
        impact = "利空（價格可能下跌）"

    # 記錄預測結果用於增量學習評估
    log_prediction_result(text, final_bias)

    return {
        "sentiment": sentiment,
        "impact": impact,
        "bias": final_bias,
        "event_risk": event_risk,
        "score": final_bias,
        "ai_bias": ai_bias,
        "ai_confidence": ai_confidence,
        "tags": tags,
        "is_event": False,
        "fusion_method": fusion_note,
    }



# 新增: 標準化新聞訊息格式 + 顯示 AI 學習進度
def build_news_message(news_text, now_time=None):
    if now_time is None:
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

    source = "News"
    if str(news_text).startswith("[CoinDesk]"):
        source = "CoinDesk"
    elif str(news_text).startswith("[Cointelegraph]"):
        source = "Cointelegraph"

    raw_text = re.sub(r"^\[[^\]]+\]\s*", "", str(news_text)).strip()
    zh_text = translate_news_to_zh(raw_text)

    # ===== AI 交易解讀 =====
    analysis = analyze_news_text(raw_text)
    sentiment = analysis["sentiment"]
    impact = analysis["impact"]
    confidence = analysis["ai_confidence"]
    bias = analysis["bias"]

    # ===== 根據利多/利空選擇對應 emoji 與標題 =====
    if bias >= 2:
        header_emoji = "🟢"
        header_label = "市場利多快訊（即時）"
    elif bias == 1:
        header_emoji = "🟢"
        header_label = "市場輕微利多快訊（即時）"
    elif bias == -1:
        header_emoji = "🔴"
        header_label = "市場輕微利空快訊（即時）"
    elif bias <= -2:
        header_emoji = "🔴"
        header_label = "市場利空快訊（即時）"
    else:
        header_emoji = "🟡"
        header_label = "市場快訊（即時）"
    
    # ===== 顯示 AI 學習狀態 =====
    accuracy_info = get_prediction_accuracy()
    accuracy_str = f"準率: {accuracy_info['accuracy']}% ({accuracy_info['correct']}/{accuracy_info['total']})" if accuracy_info['total'] > 0 else "準率: 初始化中"

    return (
        f"{header_emoji} {header_label}\n"
        f"⏰ {now_time}\n"
        f"━━━━━━━━━━━━━━\n"
        f"來源: {source}\n"
        f"📊 解讀: {sentiment}\n"
        f"🎯 置信度: {confidence:.1%}\n"
        f"🧠 {accuracy_str}\n"
        f"🔥 影響: {impact}\n"
        f"🌐 新聞(中文): {zh_text}\n"
        f"📝 原文: {raw_text}\n"
        f"━━━━━━━━━━━━━━"
    )


def _walk_strings(obj, limit=3000):
    """遞迴抽取 JSON / list / dict 內所有字串，避免 Binance 改版就整個抓不到。"""
    out = []
    q = deque([obj])

    while q and len(out) < limit:
        cur = q.popleft()

        if isinstance(cur, dict):
            for v in cur.values():
                q.append(v)
        elif isinstance(cur, list):
            for v in cur:
                q.append(v)
        elif isinstance(cur, str):
            s = normalize_news_text(cur)
            if s:
                out.append(s)

    return out


def _looks_like_news_title(text):
    low = text.lower().strip()

    bad_keywords = [
        "login", "sign up", "cookie", "terms of use", "notification center",
        "history", "creator center", "download app", "markets overview",
        "english", "discover", "bookmarks", "platform t&cs", "sitemap",
        "risk warning", "privacy", "copyright", "all rights reserved",
        "binance app", "binance feed", "latest binance news"
    ]

    if len(text) < 18 or len(text) > 220:
        return False
    if any(k in low for k in bad_keywords):
        return False
    if low.count("binance") >= 4:
        return False
    if low.startswith("/") or low.startswith("{") or low.startswith("["):
        return False
    if text.count("|") >= 4:
        return False

    # 至少要像標題，不要只是零碎 UI 字串
    has_signal_word = any(k in low for k in [
        "will", "launch", "list", "listing", "delist", "support", "upgrade",
        "airdrop", "futures", "alpha", "sec", "etf", "bitcoin", "ether",
        "crypto", "market", "fed", "cpi", "inflation", "tariff", "trump",
        "token", "partnership", "hack", "exploit", "lawsuit", "outflow",
        "inflow", "surge", "drop", "plunge", "rally", "approval",
        "fomc", "powell", "pce", "jobs", "nonfarm", "rate cut", "rate hike"
    ])

    word_count = len(text.split())
    return has_signal_word or word_count >= 5


def _extract_binance_titles_from_html(body):
    """先抓 script JSON，再退回 regex，盡量避免因頁面改版而抓不到。"""
    candidates = []

    # 1) 優先解析 script 內嵌 JSON
    script_patterns = [
        r'<script[^>]*id="__APP_DATA"[^>]*>(.*?)</script>',
        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        r'window\.__APP_DATA__\s*=\s*(\{.*?\})\s*;</script>',
        r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;</script>',
    ]

    for pattern in script_patterns:
        for raw in re.findall(pattern, body, flags=re.S):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                for s in _walk_strings(parsed):
                    if _looks_like_news_title(s):
                        candidates.append(s)
            except Exception:
                continue

    # 2) regex 後備方案
    title_patterns = [
        r'"title":"([^"\\]{12,220}(?:\\.[^"\\]*)*)"',
        r'"headline":"([^"\\]{12,220}(?:\\.[^"\\]*)*)"',
        r'"subTitle":"([^"\\]{12,220}(?:\\.[^"\\]*)*)"',
        r'>([^<>]{18,220})</a>',
        r'>([^<>]{18,220})</h3>',
        r'>([^<>]{18,220})</h2>',
    ]

    for pattern in title_patterns:
        for raw in re.findall(pattern, body, flags=re.S):
            text = normalize_news_text(raw)
            if _looks_like_news_title(text):
                candidates.append(text)

    return candidates


# === RSS/ATOM 快訊聚合 ===

def _looks_like_macro_news_title(text):
    low = text.lower().strip()

    bad_keywords = [
        "podcast", "newsletter", "advertisement", "sponsored", "opinion",
        "privacy policy", "terms of use", "sign up", "subscribe", "contact us"
    ]

    if len(text) < 18 or len(text) > 240:
        return False
    if any(k in low for k in bad_keywords):
        return False

    return any(k in low for k in [
        "bitcoin", "btc", "ether", "eth", "ethereum", "crypto", "binance",
        "sec", "etf", "fed", "fomc", "powell", "inflation", "cpi", "tariff",
        "market", "token", "listing", "delist", "hack", "exploit", "lawsuit",
        "approval", "approved", "debut", "launch", "listing", "surge", "drop",
        "plunge", "rally", "outflow", "inflow", "ceasefire", "war", "sanction",
        "bank", "banks", "digital franc", "institutional"
    ]) or len(text.split()) >= 6


def fetch_rss_news(feed_url, source_name):
    """抓 RSS / Atom，回傳 [{source, text}]，比抓 HTML 穩定很多。"""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    res = HTTP_SESSION.get(feed_url, headers=headers, timeout=8)
    res.raise_for_status()

    body = res.text.strip()
    if not body:
        return []

    results = []
    seen = set()

    try:
        root = ET.fromstring(body)
    except Exception:
        return []

    channel_items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for item in channel_items:
        title_el = item.find("title")
        title = normalize_news_text(title_el.text if title_el is not None else "")
        if not title or not _looks_like_macro_news_title(title):
            continue
        low = title.lower()
        if low in seen:
            continue
        seen.add(low)
        results.append({"source": source_name, "text": title})

    for entry in atom_entries:
        title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        title = normalize_news_text(title_el.text if title_el is not None else "")
        if not title or not _looks_like_macro_news_title(title):
            continue
        low = title.lower()
        if low in seen:
            continue
        seen.add(low)
        results.append({"source": source_name, "text": title})

    return results[:30]


def fetch_macro_rss_news():
    """聚合較穩定的 RSS / Atom 快訊來源。"""
    feeds = [
        # 1. Investing（新聞）- 替換失效鏈接
        ("https://www.investing.com/rss/news.rss", "Investing"),
        ("https://www.investing.com/rss/news_25.rss", "Investing Crypto"),
        ("https://www.investing.com/rss/news_301.rss", "Investing Commodities"),
        ("https://www.investing.com/rss/news_1.rss", "Investing Forex"),
        # 移除失效的 news_6.rss，改用更可靠的來源

        # 2. 替代新聞來源
        ("https://feeds.bloomberg.com/markets/news.rss", "Bloomberg"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC"),
        ("https://finance.yahoo.com/rss/", "Yahoo Finance"),

        # 3. 加密貨幣新聞
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
        ("https://cointelegraph.com/rss", "Cointelegraph"),

        # 4. 外匯分析 - 替換失效來源
        ("https://www.forexlive.com/feed/", "ForexLive"),
        ("https://www.investing.com/rss/forex.rss", "Technical Analysis"),
    ]

    aggregated = []
    for feed_url, source_name in feeds:
        try:
            aggregated.extend(fetch_rss_news(feed_url, source_name))
        except Exception as e:
            now_err = time.time()
            key = f"rss_err_{source_name.lower()}"
            last_err = getattr(fetch_macro_rss_news, key, 0)
            if now_err - last_err > 60:
                print(f"⚠️ {source_name} RSS error:", repr(e))
                setattr(fetch_macro_rss_news, key, now_err)

    dedup = []
    seen = set()
    for item in aggregated:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "RSS")).strip() or "RSS"
        text = normalize_news_text(item.get("text", ""))
        if not text:
            continue
        key = f"{src}|{text.lower()}"
        if key in seen:
            continue
        seen.add(key)
        dedup.append({"source": src, "text": text})

    return dedup[:50]


# 新增: 獨立刷新 RSS 新聞快取
def refresh_rss_news_cache(force=False):
    """獨立刷新 RSS 新聞快取，避免整個 macro 每 3 秒都重抓新聞。"""
    global NEWS_CACHE

    now = time.time()
    if not force and now - NEWS_CACHE.get("ts", 0) < 20:
        return NEWS_CACHE.get("news", 0), NEWS_CACHE.get("event", 0), NEWS_CACHE.get("news_list", [])

    news_bias = 0
    event_risk = 0
    news_list = NEWS_CACHE.get("news_list", [])

    try:
        aggregated_items = fetch_macro_rss_news()
        if not aggregated_items:
            NEWS_CACHE = {
                "news": NEWS_CACHE.get("news", 0),
                "event": NEWS_CACHE.get("event", 0),
                "news_list": NEWS_CACHE.get("news_list", []),
                "ts": now
            }
            return NEWS_CACHE["news"], NEWS_CACHE["event"], NEWS_CACHE["news_list"]

        if not hasattr(refresh_rss_news_cache, "seen_news"):
            refresh_rss_news_cache.seen_news = set()
        if not hasattr(refresh_rss_news_cache, "bootstrapped_news"):
            refresh_rss_news_cache.bootstrapped_news = False

        normalized_items = []
        dedup_now = set()
        for item in aggregated_items:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "News")).strip() or "News"
            text = normalize_news_text(item.get("text", ""))
            if not text:
                continue
            key = f"{src}|{text.lower()}"
            if key in dedup_now:
                continue
            dedup_now.add(key)
            normalized_items.append({"source": src, "text": text})

        news_list = []
        news_bias = 0
        event_risk = 0

        # 即使沒有新快訊，也保留近期標題供監控顯示
        latest_news = [f"[{item['source']}] {item['text'][:200]}" for item in normalized_items[:12]]

        if not refresh_rss_news_cache.bootstrapped_news:
            startup_items = normalized_items[:8]
            for item in startup_items:
                src = item["source"]
                text = item["text"]
                refresh_rss_news_cache.seen_news.add(f"{src}|{text}")
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

            for item in normalized_items[8:]:
                refresh_rss_news_cache.seen_news.add(f"{item['source']}|{item['text']}")

            refresh_rss_news_cache.bootstrapped_news = True
        else:
            for item in normalized_items:
                src = item["source"]
                text = item["text"]
                seen_key = f"{src}|{text}"
                if seen_key in refresh_rss_news_cache.seen_news:
                    continue

                refresh_rss_news_cache.seen_news.add(seen_key)
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

        # 若本輪沒有新快訊，回退為近期標題，避免監控面板長期顯示「暫無資料」
        if not news_list and latest_news:
            news_list = latest_news

        if len(refresh_rss_news_cache.seen_news) > 4000:
            refresh_rss_news_cache.seen_news = set(list(refresh_rss_news_cache.seen_news)[-2000:])

        news_bias = max(-3, min(news_bias, 3))
        event_risk = min(event_risk, 3)

        NEWS_CACHE = {
            "news": news_bias,
            "event": event_risk,
            "news_list": news_list,
            "ts": now
        }
        return news_bias, event_risk, news_list

    except Exception as e:
        last_err = getattr(refresh_rss_news_cache, "last_err_ts", 0)
        if now - last_err > 60:
            print("⚠️ RSS refresh error:", repr(e), "| use cached news")
            refresh_rss_news_cache.last_err_ts = now
        NEWS_CACHE = {
            "news": NEWS_CACHE.get("news", 0),
            "event": NEWS_CACHE.get("event", 0),
            "news_list": NEWS_CACHE.get("news_list", []),
            "ts": now
        }
        return NEWS_CACHE["news"], NEWS_CACHE["event"], NEWS_CACHE["news_list"]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ===== Binance Futures 交易輔助函數 =====

def _binance_sign(params: dict) -> dict:
    """在 params 中加入 timestamp 並附上 HMAC-SHA256 簽名。"""
    params["timestamp"] = int(time.time() * 1000)
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(
        BINANCE_API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = signature
    return params


def binance_get_position(symbol: str = "ETHUSDT") -> float:
    """回傳 Binance Futures 目前持倉數量（正數=多，負數=空，0=無倉位）。
    若 API Key 未設定則直接回傳 0.0。"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return 0.0

    try:
        params = _binance_sign({"symbol": symbol})
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        res = requests.get(
            f"{BINANCE_FAPI_BASE}/fapi/v2/positionRisk",
            params=params,
            headers=headers,
            timeout=5,
        )
        data = res.json()
        if isinstance(data, list):
            for item in data:
                if item.get("symbol") == symbol:
                    return float(item.get("positionAmt", 0.0))
        elif isinstance(data, dict):
            # 若回傳錯誤訊息
            print(f"⚠️ binance_get_position 錯誤: {data.get('msg', data)}")
    except Exception as e:
        print(f"⚠️ binance_get_position 例外: {e}")
    return 0.0


def binance_futures_market_order(
    symbol: str,
    side: str,
    quantity: float,
    reduce_only: bool = False,
) -> bool:
    """在 Binance Futures 下 MARKET 單。
    side: 'BUY' 或 'SELL'
    quantity: ETH 數量（正數，最小 0.001，精度 3 位小數）
    reduce_only: True = 減倉單（不會新開倉）
    回傳 True 表示成功。"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("⚠️ 未設定 BINANCE_API_KEY / BINANCE_API_SECRET，無法下單")
        return False

    qty = round(quantity, 3)
    if qty < _MIN_ORDER_QTY:
        print(f"⚠️ 下單數量 {qty} 過小，略過")
        return False

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": f"{qty:.3f}",
    }
    if reduce_only:
        params["reduceOnly"] = "true"

    _binance_sign(params)

    try:
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        res = requests.post(
            f"{BINANCE_FAPI_BASE}/fapi/v1/order",
            params=params,
            headers=headers,
            timeout=5,
        )
        data = res.json()
        if data.get("orderId"):
            oid = str(data["orderId"])
            print(
                f"✅ Binance 下單成功 | {side} {qty} {symbol} "
                f"orderId={oid}"
            )
            return oid
        else:
            print(f"⚠️ Binance 下單失敗: {data.get('msg', data)}")
    except Exception as e:
        print(f"⚠️ Binance 下單例外: {e}")
    return ""


def binance_set_leverage(symbol: str, leverage: int) -> bool:
    """設定 Binance Futures 槓桿倍數，回傳 True 表示成功。"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return False
    params = {"symbol": symbol, "leverage": leverage}
    _binance_sign(params)
    try:
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        res = requests.post(
            f"{BINANCE_FAPI_BASE}/fapi/v1/leverage",
            params=params,
            headers=headers,
            timeout=5,
        )
        data = res.json()
        if data.get("leverage"):
            print(f"✅ Binance 槓桿設定成功: {data['leverage']}x")
            return True
        else:
            print(f"⚠️ Binance 槓桿設定失敗: {data.get('msg', data)}")
    except Exception as e:
        print(f"⚠️ Binance 設定槓桿例外: {e}")
    return False


def binance_cancel_all_orders(symbol: str) -> bool:
    """取消指定交易對的所有掛單，回傳 True 表示成功。"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return False
    params = {"symbol": symbol}
    _binance_sign(params)
    try:
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        res = requests.delete(
            f"{BINANCE_FAPI_BASE}/fapi/v1/allOpenOrders",
            params=params,
            headers=headers,
            timeout=5,
        )
        data = res.json()
        if res.status_code == 200:
            print(f"✅ Binance 已取消所有掛單 ({symbol})")
            return True
        else:
            print(f"⚠️ Binance 取消掛單失敗: {data.get('msg', data)}")
    except Exception as e:
        print(f"⚠️ Binance 取消掛單例外: {e}")
    return False


def _binance_papi_conditional_order(
    symbol: str,
    side: str,
    strategy_type: str,
    stop_price: float,
    direction: str,
    quantity: float = 0.0,
) -> bool:
    """使用 Binance Portfolio Margin API 掛條件單（止盈/止損）。
    當標準 FAPI 端點回傳 -4120 時作為備援。
    strategy_type: 'TAKE_PROFIT' 或 'STOP'
    回傳 True 表示成功。"""
    params = {
        "symbol": symbol,
        "side": side,
        "positionSide": "LONG" if direction.lower() == "long" else "SHORT",
        "strategyType": strategy_type,
        "stopPrice": f"{stop_price:.2f}",
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }
    qty = round(quantity, 3) if quantity else 0.0
    if qty >= _MIN_ORDER_QTY:
        params["quantity"] = f"{qty:.3f}"
        params["reduceOnly"] = "true"
    else:
        params["closePosition"] = "true"

    _binance_sign(params)
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        res = requests.post(
            f"{BINANCE_PAPI_BASE}/papi/v1/um/conditional/order",
            params=params,
            headers=headers,
            timeout=5,
        )
        data = res.json()
        if data.get("strategyId"):
            print(
                f"✅ Binance PAPI {strategy_type} 掛單成功 | "
                f"stopPrice={stop_price:.2f} strategyId={data['strategyId']}"
            )
            return True
        else:
            print(f"⚠️ Binance PAPI {strategy_type} 掛單失敗: {data}")
    except Exception as e:
        print(f"⚠️ Binance PAPI {strategy_type} 掛單例外: {e}")
    return False


def binance_place_tp_sl_orders(
    symbol: str,
    direction: str,
    tp_price: float,
    sl_price: float,
    quantity: float = 0.0,
) -> bool:
    """在 Binance 使用 PAPI Algo Order 端點掛止盈（TAKE_PROFIT）與止損（STOP）條件單。
    quantity: 持倉數量，優先傳入以提升相容性。
    回傳 True 表示兩筆掛單均成功。"""
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("⚠️ 未設定 BINANCE_API_KEY / BINANCE_API_SECRET，無法掛 TP/SL")
        return False

    if tp_price <= 0 or sl_price <= 0:
        print(f"⚠️ TP/SL 價格無效（tp={tp_price}, sl={sl_price}），略過掛單")
        return False

    # long: TP 在上（賣），SL 在下（賣）
    # short: TP 在下（買），SL 在上（買）
    if direction == "long":
        tp_side = "SELL"
        sl_side = "SELL"
    else:
        tp_side = "BUY"
        sl_side = "BUY"

    tp_ok = _binance_papi_conditional_order(
        symbol, tp_side, "TAKE_PROFIT", tp_price, direction, quantity
    )
    sl_ok = _binance_papi_conditional_order(
        symbol, sl_side, "STOP", sl_price, direction, quantity
    )

    return tp_ok and sl_ok


def manage_position_scaling(current_price, atr=None):
    """持倉中的補倉/減倉管理（虛擬倉位 + 實際下單）。"""
    if not active_trade.get("open"):
        return

    now_ts = time.time()
    cooldown = 120
    add_step = 0.12
    reduce_step = 0.12
    max_add_count = 20

    last_adjust = _safe_float(active_trade.get("last_adjust_ts"), 0.0)
    if now_ts - last_adjust < cooldown:
        return

    direction = active_trade.get("direction")
    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), current_price)
    size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    max_size = _safe_float(active_trade.get("max_size"), 1 / 3)
    min_size = _safe_float(active_trade.get("min_size"), 0.15)
    add_count = int(active_trade.get("add_count", 0))

    # 以進場價附近與小幅浮盈作為調整條件，避免無限頻繁調倉
    if direction == "long":
        add_trigger = current_price <= entry * 0.997
        reduce_trigger = current_price >= entry * 1.004
    elif direction == "short":
        add_trigger = current_price >= entry * 1.003
        reduce_trigger = current_price <= entry * 0.996
    else:
        return

    # 補倉：逆勢回踩時逐步加碼（有上限）
    if add_trigger and add_count < max_add_count and size < max_size - 1e-9:
        delta = min(add_step, max_size - size)
        if delta > 0:
            new_size = size + delta
            # 均價更新（虛擬倉位）
            new_entry = ((entry * size) + (current_price * delta)) / max(new_size, 1e-9)
            tp_text = f"{_safe_float(active_trade.get('tp'), 0.0):.2f}" if active_trade.get("tp") is not None else "N/A"
            sl_text = f"{_safe_float(active_trade.get('sl'), 0.0):.2f}" if active_trade.get("sl") is not None else "N/A"
            active_trade["entry"] = float(new_entry)
            active_trade["avg_entry"] = float(new_entry)
            active_trade["size"] = float(new_size)
            active_trade["add_count"] = add_count + 1
            active_trade["last_adjust_ts"] = now_ts

            # ===== 補倉：實際向 Binance 下加倉單 =====
            if BINANCE_API_KEY and BINANCE_API_SECRET and current_price > 0:
                position_amt = abs(binance_get_position("ETHUSDT"))
                if position_amt > 0:
                    # 按倉位比例計算本次加倉數量
                    add_qty = position_amt * (delta / max(size, 1e-9))
                    if add_qty >= _MIN_ORDER_QTY:
                        add_side = "BUY" if direction == "long" else "SELL"
                        binance_futures_market_order("ETHUSDT", add_side, add_qty, reduce_only=False)
                else:
                    print("⚠️ Binance 無持倉，補倉下單略過（虛擬倉位已更新）")

            send_telegram(
                f"➕ 補倉（{direction}）\n"
                f"現價: {current_price:.2f} | 加倉: +{int(delta*100)}%\n"
                f"進場均價: {new_entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%",
                priority=True,
            )
            refresh_position_panel_from_active_trade()
            return

    # 減倉：有利方向浮盈時鎖定部分利潤（保留底倉）
    if reduce_trigger and size > min_size + 1e-9:
        delta = min(reduce_step, size - min_size)
        if delta > 0:
            new_size = size - delta
            tp_text = f"{_safe_float(active_trade.get('tp'), 0.0):.2f}" if active_trade.get("tp") is not None else "N/A"
            sl_text = f"{_safe_float(active_trade.get('sl'), 0.0):.2f}" if active_trade.get("sl") is not None else "N/A"
            active_trade["size"] = float(new_size)
            active_trade["reduce_count"] = int(active_trade.get("reduce_count", 0)) + 1
            active_trade["last_adjust_ts"] = now_ts

            # ===== 減倉：實際向 Binance 下減倉單（reduceOnly）=====
            if BINANCE_API_KEY and BINANCE_API_SECRET and current_price > 0:
                position_amt = abs(binance_get_position("ETHUSDT"))
                if position_amt > 0:
                    # 按倉位比例計算本次減倉數量
                    reduce_qty = position_amt * (delta / max(size, 1e-9))
                    if reduce_qty >= _MIN_ORDER_QTY:
                        reduce_side = "SELL" if direction == "long" else "BUY"
                        ok = binance_futures_market_order("ETHUSDT", reduce_side, reduce_qty, reduce_only=True)
                        if not ok:
                            print(f"⚠️ 減倉下單失敗，虛擬倉位仍已更新 size={new_size:.3f}")
                else:
                    print("⚠️ Binance 無持倉，減倉下單略過（虛擬倉位已更新）")

            send_telegram(
                f"➖ 減倉（{direction}）\n"
                f"現價: {current_price:.2f} | 減倉: -{int(delta*100)}%\n"
                f"進場均價: {entry:.2f} | TP: {tp_text} | SL: {sl_text}\n"
                f"倉位: {int(size*100)}% → {int(new_size*100)}%",
                priority=True,
            )
            refresh_position_panel_from_active_trade()


def maybe_decay_take_profit(current_price):
    """同一張單持倉超過 4 小時後，逐步降低止盈位。
    下修底線為進場均價加計雙邊手續費（含 50% 緩衝），確保止盈仍能獲利。"""
    if not active_trade.get("open"):
        return

    direction = active_trade.get("direction")
    if direction not in ("long", "short"):
        return

    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    tp = _safe_float(active_trade.get("tp"), 0.0)
    open_ts = _safe_float(active_trade.get("open_ts"), 0.0)
    if entry <= 0 or tp <= 0 or open_ts <= 0:
        return

    start_after = 4 * 60 * 60
    every = 60 * 60
    decay_ratio = 0.18

    held_sec = time.time() - open_ts
    if held_sec < start_after:
        return

    expected_count = int((held_sec - start_after) // every) + 1
    current_count = int(active_trade.get("tp_decay_count", 0))
    steps = expected_count - current_count
    if steps <= 0:
        return

    # 手續費底線：TP 至少要超過進場均價 + 雙邊手續費（加 50% 緩衝）
    fee_buffer = entry * ROUND_TRIP_FEE_RATE * 1.5

    old_tp = tp
    for _ in range(steps):
        if direction == "long":
            dist = max(tp - entry, 0.0)
            if dist <= 0:
                break
            tp = entry + dist * (1.0 - decay_ratio)
            # ① 不得低於手續費底線（確保止盈後實際獲利為正）
            tp = max(tp, entry + fee_buffer)
            # ② 不得低於當前價格小幅上方（避免立即觸發止盈）
            tp = max(tp, current_price * 1.0004)
        else:
            dist = max(entry - tp, 0.0)
            if dist <= 0:
                break
            tp = entry - dist * (1.0 - decay_ratio)
            # ① 不得高於手續費底線（確保止盈後實際獲利為正）
            tp = min(tp, entry - fee_buffer)
            # ② 不得高於當前價格小幅下方（避免立即觸發止盈）
            tp = min(tp, current_price * 0.9996)

    active_trade["tp"] = float(tp)
    active_trade["tp_decay_count"] = expected_count

    if abs(tp - old_tp) > 1e-9:
        hours = held_sec / 3600.0
        send_telegram(
            f"⏱️ 持倉超時下修止盈（{direction}）\n"
            f"持倉: {hours:.1f}h | 現價: {current_price:.2f}\n"
            f"TP: {old_tp:.2f} → {tp:.2f}",
            priority=True,
        )
        refresh_position_panel_from_active_trade()


def get_signal_direction(signal):
    s = str(signal or "")
    if "做多" in s:
        return "long"
    if "做空" in s:
        return "short"
    return None


def auto_fix_trade_plan(signal, entry, sl, tp, atr):
    """最終開單前修正 TP/SL，避免方向錯誤、風險距離過小，或止盈後實際收益為負（含手續費）。"""
    direction = get_signal_direction(signal)
    if direction is None:
        return signal, sl, tp

    entry = _safe_float(entry, 0.0)
    sl = _safe_float(sl, entry)
    tp = _safe_float(tp, entry)
    atr = max(_safe_float(atr, 0.0), 0.0)

    # 最小風險距離：避免 SL/TP 太近造成雜訊掃損
    min_risk = max(entry * 0.0008, atr * 0.35, 0.3)

    # 手續費最低獲利門檻：TP 必須超過開倉 + 平倉雙邊手續費，加上額外緩衝（50%）
    fee_buffer = entry * ROUND_TRIP_FEE_RATE * 1.5

    if direction == "long":
        if sl >= entry - min_risk:
            sl = entry - min_risk
        risk = max(entry - sl, min_risk)
        min_tp = entry + max(risk * 1.4, min_risk * 1.2, fee_buffer)
        if tp <= min_tp:
            tp = min_tp
    else:
        if sl <= entry + min_risk:
            sl = entry + min_risk
        risk = max(sl - entry, min_risk)
        min_tp = entry - max(risk * 1.4, min_risk * 1.2, fee_buffer)
        if tp >= min_tp:
            tp = min_tp

    return signal, float(sl), float(tp)


def get_macro_bias():
    global MACRO_CACHE

    now = time.time()

    # 🔥 低延遲模式（接近金十）
    if now - MACRO_CACHE["ts"] < 3:
        return MACRO_CACHE["sp"], MACRO_CACHE["nq"], MACRO_CACHE["btc"], MACRO_CACHE["dxy"], MACRO_CACHE.get("news", 0), MACRO_CACHE.get("event", 0), MACRO_CACHE.get("news_list", [])

    # ===== SP500 =====
    try:
        sp_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=ES=F"
        sp_data = HTTP_SESSION.get(sp_url, timeout=3).json()
        sp_price = float(sp_data["quoteResponse"]["result"][0]["regularMarketPrice"])
        sp_prev = float(sp_data["quoteResponse"]["result"][0]["regularMarketPreviousClose"])
        sp_change = (sp_price - sp_prev) / sp_prev
    except:
        sp_change = 0

    # ===== NASDAQ (NQ) =====
    try:
        nq_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=NQ=F"
        nq_data = HTTP_SESSION.get(nq_url, timeout=3).json()
        nq_price = float(nq_data["quoteResponse"]["result"][0]["regularMarketPrice"])
        nq_prev = float(nq_data["quoteResponse"]["result"][0]["regularMarketPreviousClose"])
        nq_change = (nq_price - nq_prev) / nq_prev
    except:
        nq_change = 0

    # ===== BTC（相對性核心）=====
    try:
        btc_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        btc_data = HTTP_SESSION.get(btc_url, timeout=3).json()
        btc_change = float(btc_data["priceChangePercent"]) / 100
    except:
        btc_change = 0

    # ===== DXY =====
    try:
        dxy_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=DX-Y.NYB"
        dxy_data = HTTP_SESSION.get(dxy_url, timeout=3).json()
        dxy_price = float(dxy_data["quoteResponse"]["result"][0]["regularMarketPrice"])
        dxy_prev = float(dxy_data["quoteResponse"]["result"][0]["regularMarketPreviousClose"])
        dxy_change = (dxy_price - dxy_prev) / dxy_prev
    except:
        dxy_change = 0

    # ===== 即時新聞聚合（僅 RSS，已移除 Binance / Jin10） =====
    news_bias, event_risk, news_list = refresh_rss_news_cache(force=False)

    MACRO_CACHE = {"sp": sp_change, "nq": nq_change, "btc": btc_change, "dxy": dxy_change, "news": news_bias, "event": event_risk, "news_list": news_list, "ts": now}
    return sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk, news_list

# ===== Online Model Persistence =====
ONLINE_MODEL_PATH = "/Volumes/SSD/trading/online_model.pkl"

# =============================
# 全域
# =============================
WS_LOCK = threading.Lock()
WS_PRICE = None

# ===== 勝率統計 =====
performance = {
    "total": 0,
    "win": 0,
    "loss": 0
}

# ===== 交易狀態（真實交易管理） =====
active_trade = {
    "direction": None,
    "entry": None,
    "avg_entry": None,
    "tp": None,
    "sl": None,
    "open": False,
    "size": 0.0,
    "max_size": 1 / 3,
    "min_size": 0.15,
    "add_count": 0,
    "reduce_count": 0,
    "last_adjust_ts": 0.0,
    "open_ts": 0.0,
    "tp_decay_count": 0,
}

# =============================
# KLINE CACHE（避免打爆API）
# =============================
KLINE_CACHE = {}
KLINE_TTL = {
    "4h": 60*60,
    "1h": 60*10,
    "30m": 60*5,
    "15m": 60*3,
    "5m": 60*2,
    "1m": 10
}

# =============================
# WebSocket（tick級）
# =============================
def ws_price_stream():
    def on_message(ws, msg):
        global WS_PRICE
        data = json.loads(msg)
        WS_PRICE = float(data["p"])

    ws = websocket.WebSocketApp(
        "wss://fstream.binance.com/ws/ethusdt@aggTrade",
        on_message=on_message
    )

    while True:
        try:
            ws.run_forever()
        except:
            time.sleep(2)

threading.Thread(target=ws_price_stream, daemon=True).start()

# =============================
# Indicators
# =============================
def calc_indicators(df):
    df["ma25"] = df["close"].rolling(25).mean()

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()

    # ===== Volume v2 =====
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    # VWAP（簡化版：以收盤加權）
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    return df

# =============================
# Market Regime（市場狀態）
# =============================
def detect_market_regime(df_1h, df_4h):
    trend_1h = df_1h["close"].iloc[-1] - df_1h["ma25"].iloc[-1]
    trend_4h = df_4h["close"].iloc[-1] - df_4h["ma25"].iloc[-1]

    # ===== 4H 強度（新增）=====
    strength_4h = df_4h["ma25"].iloc[-1] - df_4h["ma25"].iloc[-5]

    # ===== 波動（用1H判斷市場活躍度）=====
    vol = (df_1h["high"].iloc[-1] - df_1h["low"].iloc[-1]) / df_1h["close"].iloc[-1]

    # ===== v2 分類（強弱趨勢）=====
    # 多頭
    if trend_4h > 0:
        if strength_4h > 0 and vol > 0.008:
            return "bull_trend_strong"
        return "bull_trend"

    # 空頭
    if trend_4h < 0:
        if strength_4h < 0 and vol > 0.008:
            return "bear_trend_strong"
        return "bear_trend"

    return "range"

# =============================
# FVG
# =============================
def calc_fvg(df):
    if len(df) < 3:
        return None, None

    for i in range(len(df)-2, 1, -1):
        c1 = df.iloc[i-2]
        c3 = df.iloc[i]

        if c3["low"] > c1["high"]:
            return c1["high"], c3["low"]

        if c3["high"] < c1["low"]:
            return c3["high"], c1["low"]

    return None, None

# =============================
# Triangle Pattern（三角收斂）
# =============================
def detect_triangle(df, lookback=20):
    if len(df) < lookback:
        return 0

    highs = df["high"].tail(lookback).values
    lows = df["low"].tail(lookback).values

    # 線性回歸斜率
    x = np.arange(len(highs))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]

    # 上下降、下上升 → 收斂
    if high_slope < 0 and low_slope > 0:
        return 1   # 三角收斂

    return 0

# =============================
# AI（Meta Model）
# =============================
MODEL_PATH = "/Volumes/SSD/trading/model.pkl"
DATA_PATH = "/Volumes/SSD/trading/ai_data.csv"
model = None
# Online learning model
online_model = SGDClassifier(loss="log_loss")
online_initialized = False

def load_model():
    global model, online_model, online_initialized

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        try:
            model = pickle.load(open(MODEL_PATH, "rb"))
        except Exception as e:
            print(f"⚠️ 加載模型失敗: {e}")
            model = None

    if os.path.exists(ONLINE_MODEL_PATH):
        try:
            online_model = pickle.load(open(ONLINE_MODEL_PATH, "rb"))
            online_initialized = True
        except:
            print("⚠️ 舊模型不相容，重置 online_model")
            online_model = SGDClassifier(loss="log_loss")
            online_initialized = False

    # 加載新聞模型
    load_news_model()

def update_online_model(features, label):
    global online_model, online_initialized

    X = pd.DataFrame([features])
    y = np.array([label])

    # ===== FIX: 強制對齊 feature columns =====
    if online_initialized and hasattr(online_model, "feature_names_in_"):
        expected_cols = list(online_model.feature_names_in_)

        # 補缺的欄位
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0

        # 移除多餘欄位
        X = X[expected_cols]

    try:
        if not online_initialized:
            online_model.partial_fit(X, y, classes=np.array([0, 1]))
            online_initialized = True
        else:
            online_model.partial_fit(X, y)
    except Exception as e:
        print("⚠️ online_model error, reset model:", e)
        online_model = SGDClassifier(loss="log_loss")
        online_model.partial_fit(X, y, classes=np.array([0, 1]))
        online_initialized = True

    # persist online model
    def _save():
        try:
            with open(ONLINE_MODEL_PATH, "wb") as f:
                pickle.dump(online_model, f)
        except:
            pass

    threading.Thread(target=_save, daemon=True).start()

def train_model():
    global model
    if not os.path.exists(DATA_PATH):
        return

    df = pd.read_csv(DATA_PATH)
    if len(df) < 50:
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(n_estimators=120)
    model.fit(X, y)

    def _save_model():
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
        except:
            pass
    threading.Thread(target=_save_model, daemon=True).start()
    print("✅ AI 更新")

def retrain_model():
    """強制重新訓練 AI 模型"""
    global model
    print("🔄 開始重新訓練 AI 模型...")

    if not os.path.exists(DATA_PATH):
        print("⚠️ 沒有訓練數據檔案")
        return

    try:
        df = pd.read_csv(DATA_PATH, header=None)
        # 數據沒有標題，直接使用最後一列作為 label
        df.columns = [f"feature_{i}" for i in range(len(df.columns) - 1)] + ["label"]
    except Exception as e:
        print(f"⚠️ 讀取訓練數據失敗: {e}")
        return

    if len(df) < 10:
        print("⚠️ 訓練數據不足（至少需要10筆）")
        return

    # 只使用最近的1000筆數據來加速訓練
    if len(df) > 1000:
        df = df.tail(1000)
        print(f"✅ 使用最近1000筆數據訓練")

    # 確保有 label 列
    if "label" not in df.columns:
        print("⚠️ 數據缺少 label 列")
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print("✅ AI 模型重新訓練完成並保存")
    except Exception as e:
        print(f"⚠️ 保存模型失敗: {e}")

log_buffer = []

def log_data(features, label):
    global log_buffer

    log_buffer.append({**features, "label": label})

    if len(log_buffer) >= 20:
        df = pd.DataFrame(log_buffer)

        if os.path.exists(DATA_PATH):
            df.to_csv(DATA_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(DATA_PATH, index=False)

        log_buffer = []

def send_telegram(msg, priority=False, pin=False):
    global LAST_TELEGRAM_TS, TELEGRAM_PINNED_MESSAGE_ID

    now = time.time()

    # ===== 只有低優先才限流 =====
    if not priority and now - LAST_TELEGRAM_TS < 10:
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram 未設定，略過發送")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    # V7 安全版 + 避免特殊字元炸掉
    safe_msg = str(msg).replace("<", "").replace(">", "").replace("&", "and")

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": safe_msg
    }

    try:
        res = requests.post(url, data=payload, timeout=5)
        if res.status_code == 400:
            # fallback without any特殊字元問題
            payload["text"] = str(msg).replace("<", "").replace(">", "")
            res = requests.post(url, data=payload, timeout=5)
        sent_message_id = None

        if res.status_code != 200:
            print("❌ Telegram 發送失敗:", res.status_code, res.text)
        else:
            print("✅ Telegram 已送出")
            try:
                body = res.json()
                sent_message_id = body.get("result", {}).get("message_id")
            except Exception:
                sent_message_id = None

        # ===== retry（避免偶發失敗） =====
        if res.status_code != 200:
            try:
                time.sleep(1)
                res2 = requests.post(url, data=payload, timeout=5)
                print("🔁 retry:", res2.status_code)
            except Exception as e:
                print("❌ retry失敗:", e)

        # Discord只發「進場通知」
        try:
            if DISCORD_WEBHOOK and "進場" in msg:
                requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=5)
        except Exception as e:
            print("Discord error:", e)

        if pin and sent_message_id is not None:
            try:
                pin_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/pinChatMessage"
                pin_res = requests.post(
                    pin_url,
                    data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "message_id": sent_message_id,
                        "disable_notification": True,
                    },
                    timeout=5,
                )
                if pin_res.status_code == 200:
                    TELEGRAM_PINNED_MESSAGE_ID = sent_message_id
            except Exception as e:
                print("⚠️ Telegram 置頂失敗:", e)

        LAST_TELEGRAM_TS = now

    except Exception as e:
        print("❌ Telegram error:", e, "| msg:", msg[:50])


WEBAPP_BASE_URL = "https://josh940085.github.io/ETH-bot/"


def send_position_keyboard(direction, entry, tp, sl, size, entry_display=None, tp_display=None, sl_display=None, is_update=False):
    """進場後在 Telegram 發出倉位面板按鈕（私聊用 Web App，群組/頻道用 URL 按鈕）。
    entry_display, tp_display, sl_display: 若提供則使用此字串確保訊息與網址一致。"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        dir_param  = "long" if direction == "long" else "short"
        size_pct   = int(float(size) * 100)
        
        # 使用傳入的顯示價格，確保與訊息一致
        entry_str = entry_display if entry_display else f"{entry:.2f}"
        tp_str = tp_display if tp_display else (f"{tp:.2f}" if tp is not None else "0.0")
        sl_str = sl_display if sl_display else (f"{sl:.2f}" if sl is not None else "0.0")
        
        url = (
            f"{WEBAPP_BASE_URL}"
            f"?dir={dir_param}"
            f"&entry={entry_str}"
            f"&tp={tp_str}"
            f"&sl={sl_str}"
            f"&size={size_pct}"
            f"&pair=ETHUSDT"
            f"&lev=10"
        )
        
        # 判斷是否為私聊（chat_id 為正數）或群組/頻道（負數）
        try:
            chat_id_int = int(TELEGRAM_CHAT_ID)
            is_private = chat_id_int > 0
        except (ValueError, TypeError):
            is_private = True  # 預設為私聊
        
        if is_private:
            # 私聊：使用 Web App 按鈕（底部按鈕）
            keyboard = {
                "keyboard": [[{"text": "📊 開啟倉位面板", "web_app": {"url": url}}]],
                "resize_keyboard": True,
                "persistent": True,
            }
            text = "📊 倉位面板已更新，點擊底部按鈕查看最新數據" if is_update else "📊 倉位已建立，點擊底部按鈕查看即時面板"
        else:
            # 群組/頻道：使用 inline 按鈕（URL）
            keyboard = {
                "inline_keyboard": [[
                    {"text": "📊 開啟倉位面板", "url": url}
                ]]
            }
            text = "📊 倉位面板已更新，點擊按鈕查看最新數據" if is_update else "📊 倉位已建立，點擊按鈕查看即時面板"
        
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "reply_markup": keyboard,
            },
            timeout=5,
        )
    except Exception as e:
        print(f"⚠️ 倉位面板按鈕發送失敗: {e}")


def refresh_position_panel_from_active_trade():
    """依照目前 active_trade 狀態重送面板按鈕，保持 URL 參數與交易狀態同步。"""
    if not active_trade.get("open"):
        return

    direction = active_trade.get("direction")
    if direction not in ("long", "short"):
        return

    entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
    size = max(0.0, _safe_float(active_trade.get("size"), 0.0))
    tp = active_trade.get("tp")
    sl = active_trade.get("sl")

    if entry <= 0 or size <= 0:
        return

    entry_str = f"{entry:.2f}"
    tp_str = f"{_safe_float(tp, 0.0):.2f}" if tp is not None else "0.0"
    sl_str = f"{_safe_float(sl, 0.0):.2f}" if sl is not None else "0.0"

    send_position_keyboard(
        direction,
        entry,
        tp,
        sl,
        size,
        entry_display=entry_str,
        tp_display=tp_str,
        sl_display=sl_str,
        is_update=True,
    )


def remove_position_keyboard():
    """平倉後移除底部 Web App 鍵盤。"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": "📋 倉位已平倉，面板已關閉",
                "reply_markup": {"remove_keyboard": True},
            },
            timeout=5,
        )
    except Exception as e:
        print(f"⚠️ 移除鍵盤失敗: {e}")


def clear_telegram_pin():
    """解除目前開倉通知置頂（若有）。"""
    global TELEGRAM_PINNED_MESSAGE_ID
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if TELEGRAM_PINNED_MESSAGE_ID is None:
        return

    try:
        unpin_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/unpinChatMessage"
        requests.post(
            unpin_url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "message_id": TELEGRAM_PINNED_MESSAGE_ID,
            },
            timeout=5,
        )
    except Exception as e:
        print("⚠️ Telegram 解除置頂失敗:", e)
    finally:
        TELEGRAM_PINNED_MESSAGE_ID = None


# ===== AI分析（OpenClaw / OpenAI） =====
OPENAI_API_KEY = _get_required_env("OPENAI_API_KEY", "", mask=True)

def ask_ai_analysis(prompt):
    if not OPENAI_API_KEY:
        return "AI分析失敗: 未設定 OPENAI_API_KEY"

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "你是一個專業ETH交易分析師"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10).json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI分析失敗: {e}"


def load_last_update_id():
    try:
        if not TELEGRAM_STATE_PATH.exists():
            return None
        payload = json.loads(TELEGRAM_STATE_PATH.read_text(encoding="utf-8"))
        value = payload.get("last_update_id")
        return int(value) if value is not None else None
    except Exception:
        return None


def save_last_update_id(update_id):
    try:
        payload = {}
        if TELEGRAM_STATE_PATH.exists():
            try:
                data = json.loads(TELEGRAM_STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    payload = data
            except Exception:
                payload = {}

        payload["last_update_id"] = int(update_id)
        TELEGRAM_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def request_supervisor_restart():
    try:
        payload = {}
        if TELEGRAM_STATE_PATH.exists():
            try:
                data = json.loads(TELEGRAM_STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    payload = data
            except Exception:
                payload = {}

        payload["restart_requested"] = True
        payload["restart_requested_at"] = int(time.time())
        TELEGRAM_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        return True
    except Exception:
        return False


def pop_pending_commands():
    try:
        if not TELEGRAM_STATE_PATH.exists():
            return []

        payload = json.loads(TELEGRAM_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []

        pending = payload.get("pending_commands")
        commands = pending if isinstance(pending, list) else []
        payload["pending_commands"] = []
        TELEGRAM_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        return commands
    except Exception:
        return []


# ===== Telegram 指令（AI分析） =====
def handle_ai_command(text, context=None):
    global BOT_SOFT_RESTART_REQUESTED

    if text.startswith("/restart"):
        if os.getenv("BOT_SUPERVISOR") == "1":
            if request_supervisor_restart():
                return "♻️ 已收到 /restart，將由 program.py 執行同步並重啟。"
            return "⚠️ /restart 失敗：無法寫入重啟請求。"

        BOT_SOFT_RESTART_REQUESTED = True
        return "♻️ 已收到 /restart，將在本程序內執行軟重啟。"

    if text.startswith("/ai"):
        question = text.replace("/ai", "").strip()

        if context is None:
            context = {}

        # 如果沒輸入問題 → 自動分析當前市場
        if not question:
            question = "請根據以下數據判斷是否應該做多或做空，並給出理由與策略"

        # 注入你系統數據（核心升級）
        prompt = f"""
你是一個專業ETH交易分析師，請根據以下即時市場數據進行分析：

【市場數據】
價格: {context.get('price')}
AI分數: {context.get('score')}
HTF趨勢: {context.get('htf')}
市場狀態: {context.get('regime')}
Breakout: {context.get('breakout')}
Triangle: {context.get('triangle')}
Macro: {context.get('macro')}
Volume Spike: {context.get('volume_spike')}

【問題】
{question}

請輸出：
1. 當前市場結構
2. 是否建議做多/做空/觀望
3. 進場區間
4. 止盈止損建議
"""

        return ask_ai_analysis(prompt)

    if text.startswith("/news"):
        try:
            _, _, news_list = refresh_rss_news_cache(force=True)
            if news_list:
                preview = "\n".join([f"- {n}" for n in news_list[:12]])
                return f"📰 最新即時訊息\n{preview}"
            return "📰 目前沒有抓到新的即時訊息"
        except Exception as e:
            return f"📰 新聞讀取失敗: {e}"

    return None

load_model()

# =============================
# API（簡化 + CACHE）
# =============================
def get_kline(interval, limit=100):
    now = time.time()

    if interval in KLINE_CACHE:
        data, ts = KLINE_CACHE[interval]
        if now - ts < KLINE_TTL.get(interval, 10):
            return data

    url = "https://fapi.binance.com/fapi/v1/klines"
    data = requests.get(url, params={
        "symbol": "ETHUSDT",
        "interval": interval,
        "limit": limit
    }).json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "_","_","_","_","_","_"
    ])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    df = calc_indicators(df)

    KLINE_CACHE[interval] = (df, now)
    return df

# =============================
# 主邏輯（AI接管）
# =============================
def run_bot():
    global performance, BOT_SOFT_RESTART_REQUESTED, KLINE_CACHE, MACRO_CACHE, NEWS_CACHE

    load_model()  # 加載所有模型，包括新聞模型
    retrain_model()  # 啟動時重新訓練 AI 模型

    last_signal = None
    last_trade_time = 0
    TRADE_COOLDOWN = 300  # 冷卻加長（防洗單）
    SL_COOLDOWN = 60      # 止損後縮短冷卻時間，避免開單頻率過低
    MIN_PRICE_CHANGE = 0.002  # 至少0.2%價格變動才允許新單
    MIN_SIGNAL_DIFF = 0.05  # 信號差異門檻
    last_trade_signal = None  # 避免同一訊號重複開單
    losing_streak = 0
    MAX_LOSS_STREAK = 3
    last_entry_price = None
    last_direction = None
    # trade_open 移除，改用 active_trade 控制是否可開單

    # ===== 每日報告 =====
    last_report_time = 0

    last_update_id = load_last_update_id()

    # ===== V7 防洗單（訊號記憶）=====
    last_signal_cache = None

    while True:
        try:
            if BOT_SOFT_RESTART_REQUESTED:
                BOT_SOFT_RESTART_REQUESTED = False
                KLINE_CACHE.clear()
                MACRO_CACHE = {"sp": 0, "nq": 0, "btc": 0, "dxy": 0, "news": 0, "event": 0, "news_list": [], "ts": 0}
                NEWS_CACHE = {"news": 0, "event": 0, "news_list": [], "ts": 0}
                last_signal = None
                last_trade_time = 0
                last_trade_signal = None
                last_entry_price = None
                last_direction = None
                last_signal_cache = None
                load_model()
                print("♻️ 已完成軟重啟：模型重載、快取清空")

            # ===== Telegram 指令接收 =====
            if os.getenv("BOT_SUPERVISOR") == "1":
                updates = pop_pending_commands()
            else:
                params = {"timeout": 5}
                if last_update_id:
                    params["offset"] = last_update_id + 1

                try:
                    res = requests.get(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                        params=params,
                        timeout=6
                    )
                    updates = res.json().get("result", [])
                except:
                    updates = []

            for u in updates:
                if os.getenv("BOT_SUPERVISOR") == "1":
                    text = u.get("text", "")
                    chat_id = u.get("chat_id")
                    last_update_id = u.get("update_id", last_update_id)
                    if last_update_id is not None:
                        save_last_update_id(last_update_id)
                else:
                    last_update_id = u.get("update_id")
                    if last_update_id is not None:
                        save_last_update_id(last_update_id)
                    text = u.get("message", {}).get("text", "")
                    chat_id = u.get("message", {}).get("chat", {}).get("id")

                try:
                    if not text:
                        continue

                    # AI / 新聞指令
                    context = {
                        "price": price if 'price' in locals() else None,
                        "score": score if 'score' in locals() else None,
                        "htf": htf if 'htf' in locals() else None,
                        "regime": regime if 'regime' in locals() else None,
                        "breakout": breakout if 'breakout' in locals() else None,
                        "triangle": triangle if 'triangle' in locals() else None,
                        "macro": macro_bias if 'macro_bias' in locals() else None,
                        "volume_spike": volume_spike if 'volume_spike' in locals() else None,
                    }

                    reply = handle_ai_command(text, context)

                    if reply:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            data={"chat_id": chat_id, "text": reply},
                            timeout=5
                        )
                except:
                    pass
            # ===== HTF（方向）=====
            df_4h = get_kline("4h")
            df_1h = get_kline("1h")

            # ===== Market Regime =====
            regime = detect_market_regime(df_1h, df_4h)

            # ===== 4H v2（方向 + 強度）=====
            trend_4h = df_4h["close"].iloc[-1] - df_4h["ma25"].iloc[-1]
            strength_4h = df_4h["ma25"].iloc[-1] - df_4h["ma25"].iloc[-5]

            htf = 1 if trend_4h > 0 else -1
            htf_strength = abs(strength_4h)

            # ===== MID（策略）=====
            df_30m = get_kline("30m")
            df_15m = get_kline("15m")

            mid_trend = 1 if df_30m["macd"].iloc[-1] > df_30m["signal"].iloc[-1] else -1
            fvg_low, fvg_high = calc_fvg(df_15m)

            # ===== LTF（進場）=====
            df_5m = get_kline("5m", 50)
            df_1m = get_kline("1m", 50)

            breakout = 0
            recent_high = df_5m["high"].iloc[-5:-1].max()
            recent_low = df_5m["low"].iloc[-5:-1].min()
            price = WS_PRICE if WS_PRICE else df_1m["close"].iloc[-1]

            # ===== Macro（時事）=====
            sp_change, nq_change, btc_change, dxy_change, news_bias, event_risk, news_list = get_macro_bias()

            # ===== 🔥 即時新聞推送（任何時候都推送，不依賴是否持倉）=====
            if not hasattr(run_bot, "last_news_set"):
                run_bot.last_news_set = set()

            if not hasattr(run_bot, "startup_news_snapshot_sent"):
                run_bot.startup_news_snapshot_sent = False

            if news_list:
                new_news = []

                for n in news_list:
                    if n and n not in run_bot.last_news_set:
                        new_news.append(n)

                if not run_bot.startup_news_snapshot_sent and news_list:
                    snapshot_header = (
                        "🧭 啟動新聞快照\n"
                        "━━━━━━━━━━━━━━\n"
                        f"已抓到 {len(news_list)} 則 RSS 即時訊息\n"
                        "━━━━━━━━━━━━━━"
                    )
                    print("\n" + snapshot_header)
                    send_telegram(snapshot_header, priority=True)
                    run_bot.startup_news_snapshot_sent = True

                new_news = new_news[:15]

                if new_news:
                    now_time = datetime.datetime.now().strftime("%H:%M:%S")

                    for n in new_news:
                        msg_news = build_news_message(n, now_time)

                        # 🔥 過濾中性新聞
                        if "📊 解讀: 中性" in msg_news:
                            continue

                        print("\n" + msg_news)
                        send_telegram(msg_news, priority=True)
                        run_bot.last_news_set.add(n)

                    if len(run_bot.last_news_set) > 500:
                        run_bot.last_news_set = set(list(run_bot.last_news_set)[-200:])

            # ===== 即時新聞摘要（顯示在主訊號內）=====
            news_text = ""
            if news_list:
                news_text = "📰 重點新聞:\n"
                for n in news_list[:5]:
                    raw_item = str(n)
                    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", raw_item)
                    if m:
                        src = m.group(1).strip()
                        body = m.group(2).strip()
                    else:
                        src = "News"
                        body = raw_item

                    zh_body = translate_news_to_zh(body)
                    preview = zh_body[:100] + ("..." if len(zh_body) > 100 else "")
                    news_text += f"- {preview}\n"

            # ===== 真實交易管理（TP/SL） =====
            if active_trade["open"]:
                current = price
                candle_high = float(df_1m["high"].iloc[-1]) if len(df_1m) > 0 else current
                candle_low = float(df_1m["low"].iloc[-1]) if len(df_1m) > 0 else current

                # 同一張單持倉超過 4 小時，自動下修 TP
                maybe_decay_take_profit(current)

                if active_trade["direction"] == "long":
                    tp_hit = (current >= active_trade["tp"]) or (candle_high >= active_trade["tp"])
                    sl_hit = (current <= active_trade["sl"]) or (candle_low <= active_trade["sl"])

                    # 同根K同時觸發時採保守：先算SL，避免回測偏樂觀
                    if sl_hit:
                        performance["loss"] += 1
                        performance["total"] += 1
                        binance_cancel_all_orders("ETHUSDT")
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        clear_telegram_pin()
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak += 1
                        # 止損後重置防洗單記憶，改用短冷卻時間，避免開單頻率過低
                        last_entry_price = None
                        last_trade_signal = None
                        last_signal = None
                        last_trade_time = time.time() - (TRADE_COOLDOWN - SL_COOLDOWN)
                        print("❌ SL 命中")
                        send_telegram(
                            f"❌ SL 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

                    elif tp_hit:
                        performance["win"] += 1
                        performance["total"] += 1
                        tp_exit = active_trade["tp"]
                        avg_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), tp_exit)
                        gross_pct = (tp_exit - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                        net_pct = gross_pct - ROUND_TRIP_FEE_RATE
                        binance_cancel_all_orders("ETHUSDT")
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        clear_telegram_pin()
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak = 0
                        print("✅ TP 命中")
                        send_telegram(
                            f"✅ TP 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"進場均價: {avg_entry:.2f} | TP: {tp_exit:.2f}\n"
                            f"毛利: {gross_pct*100:+.3f}% | 手續費: -{ROUND_TRIP_FEE_RATE*100:.3f}% | 淨利: {net_pct*100:+.3f}%\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

                elif active_trade["direction"] == "short":
                    tp_hit = (current <= active_trade["tp"]) or (candle_low <= active_trade["tp"])
                    sl_hit = (current >= active_trade["sl"]) or (candle_high >= active_trade["sl"])

                    # 同根K同時觸發時採保守：先算SL，避免回測偏樂觀
                    if sl_hit:
                        performance["loss"] += 1
                        performance["total"] += 1
                        binance_cancel_all_orders("ETHUSDT")
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        clear_telegram_pin()
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak += 1
                        # 止損後重置防洗單記憶，改用短冷卻時間，避免開單頻率過低
                        last_entry_price = None
                        last_trade_signal = None
                        last_signal = None
                        last_trade_time = time.time() - (TRADE_COOLDOWN - SL_COOLDOWN)
                        print("❌ SL 命中")
                        send_telegram(
                            f"❌ SL 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

                    elif tp_hit:
                        performance["win"] += 1
                        performance["total"] += 1
                        tp_exit = active_trade["tp"]
                        avg_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), tp_exit)
                        gross_pct = (avg_entry - tp_exit) / avg_entry if avg_entry > 0 else 0.0
                        net_pct = gross_pct - ROUND_TRIP_FEE_RATE
                        binance_cancel_all_orders("ETHUSDT")
                        active_trade["open"] = False
                        active_trade["size"] = 0.0
                        active_trade["add_count"] = 0
                        active_trade["reduce_count"] = 0
                        active_trade["open_ts"] = 0.0
                        active_trade["tp_decay_count"] = 0
                        clear_telegram_pin()
                        remove_position_keyboard()
                        last_signal_cache = None
                        losing_streak = 0
                        print("✅ TP 命中")
                        send_telegram(
                            f"✅ TP 命中（{active_trade['direction']}）\n"
                            f"當前: {current:.2f} | 1m高低: {candle_high:.2f}/{candle_low:.2f}\n"
                            f"進場均價: {avg_entry:.2f} | TP: {tp_exit:.2f}\n"
                            f"毛利: {gross_pct*100:+.3f}% | 手續費: -{ROUND_TRIP_FEE_RATE*100:.3f}% | 淨利: {net_pct*100:+.3f}%\n"
                            f"已關閉倉位，等待下一筆交易",
                            priority=True
                        )

            # 命中止盈止損前提下，持倉中允許補倉/減倉
            if active_trade["open"]:
                manage_position_scaling(current)

            # ===== 核心限制：未平倉禁止開新單，但新聞照常推 =====
            if active_trade["open"]:
                if not hasattr(run_bot, "last_position_status_ts"):
                    run_bot.last_position_status_ts = 0
                if not hasattr(run_bot, "last_news_monitor_ts"):
                    run_bot.last_news_monitor_ts = 0

                if time.time() - run_bot.last_position_status_ts > 15:
                    monitor_entry = _safe_float(active_trade.get("avg_entry", active_trade.get("entry")), 0.0)
                    print(
                        f"📡 持倉監控 | 方向: {active_trade['direction']} | 倉位: {int(_safe_float(active_trade.get('size'), 0)*100)}% | 現價: {price:.2f} | 進場均價: {monitor_entry:.2f} | TP: {active_trade['tp']:.2f} | SL: {active_trade['sl']:.2f}"
                    )
                    run_bot.last_position_status_ts = time.time()

                if time.time() - run_bot.last_news_monitor_ts > 30:
                    latest_news_preview = " | ".join(news_list[:4]) if news_list else "目前無新快訊（RSS 暫無資料）"
                    print(f"📰 新聞監控中 | {latest_news_preview}")
                    run_bot.last_news_monitor_ts = time.time()

                time.sleep(0.8)
                continue

            if price > recent_high:
                breakout = 1
            elif price < recent_low:
                breakout = -1

            # ===== Liquidity Sweep（掃流動性 v7）=====
            sweep_high = False
            sweep_low = False

            prev_high = df_5m["high"].iloc[-2]
            prev_low = df_5m["low"].iloc[-2]

            # 掃上方流動性（假突破上影）
            if price > recent_high and df_5m["close"].iloc[-1] < prev_high:
                sweep_high = True

            # 掃下方流動性（假跌破下影）
            if price < recent_low and df_5m["close"].iloc[-1] > prev_low:
                sweep_low = True

            macro_bias = 0

            # BTC（最高權重）
            if btc_change > 0.002:
                macro_bias += 1.5
            elif btc_change < -0.002:
                macro_bias -= 1.5

            # ===== Macro v2（權重模型）=====
            # NASDAQ（權重最高）
            if nq_change > 0.0015:
                macro_bias += 1.2
            elif nq_change < -0.0015:
                macro_bias -= 1.2

            # SP500（輔助）
            if sp_change > 0.0015:
                macro_bias += 0.6
            elif sp_change < -0.0015:
                macro_bias -= 0.6

            # DXY（反向）
            if dxy_change > 0.0015:
                macro_bias -= 1
            elif dxy_change < -0.0015:
                macro_bias += 1

            # NEWS（新增）
            macro_bias += news_bias * 0.8
            # ===== 事件風險（波動放大器）=====
            if event_risk >= 1:
                macro_bias *= 1.2
            if event_risk >= 2:
                macro_bias *= 1.5

            # ===== Feature（升級版）=====
            recent_high = df_15m["high"].tail(20).max()
            recent_low = df_15m["low"].tail(20).min()
            triangle = detect_triangle(df_15m)

            # ===== Volume v2（量價分析）=====
            vol_now = df_15m["volume"].iloc[-1]
            vol_ma = df_15m["vol_ma20"].iloc[-1] if "vol_ma20" in df_15m.columns else df_15m["volume"].rolling(20).mean().iloc[-1]

            volume_spike = vol_now > vol_ma * 1.5
            volume_ratio = vol_now / (vol_ma + 1e-9)

            # 買賣壓（K線方向近似）
            buy_pressure = df_15m["close"].iloc[-1] > df_15m["open"].iloc[-1]
            sell_pressure = df_15m["close"].iloc[-1] < df_15m["open"].iloc[-1]

            # 吸籌 / 出貨（簡化：放量但不延續）
            prev_close = df_15m["close"].iloc[-2]
            absorption = False
            if volume_spike:
                if buy_pressure and price < prev_close:
                    absorption = True  # 出貨
                if sell_pressure and price > prev_close:
                    absorption = True  # 吸籌

            features = {
                "htf": htf,
                "htf_strength": htf_strength,
                "mid_trend": mid_trend,
                "macd": df_15m["macd"].iloc[-1],
                "hist": df_15m["macd"].iloc[-1] - df_15m["signal"].iloc[-1],
                "price_vs_ma": df_15m["close"].iloc[-1] - df_15m["ma25"].iloc[-1],
                "breakout": breakout,
                "fvg": (fvg_high - fvg_low) if fvg_low else 0,

                # 新增核心特徵
                "volatility": df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1],
                "trend_strength": abs(df_15m["ma25"].iloc[-1] - df_15m["ma25"].iloc[-5]),
                "range_pos": (price - recent_low) / (recent_high - recent_low + 1e-6),

                "sp": sp_change,
                "nq": nq_change,
                "btc": btc_change,
                "dxy": dxy_change,
                "macro": macro_bias,
                "regime": {
                    "bull_trend_strong": 2,
                    "bull_trend": 1,
                    "range": 0,
                    "bear_trend": -1,
                    "bear_trend_strong": -2
                }[regime],
                "triangle": triangle,
                "event_risk": event_risk,
                "volume_spike": int(volume_spike),
                "volume_ratio": volume_ratio,
                "buy_pressure": int(buy_pressure),
                "absorption": int(absorption),
                "sweep_high": int(sweep_high),
                "sweep_low": int(sweep_low),
            }

            # ===== AI決策（強化版：避免卡0.5）=====
            ai_prob = 0.5

            try:
                X = pd.DataFrame([features])

                # Online model（主模型）
                if online_initialized:
                    ai_prob = online_model.predict_proba(X)[0][1]

                # 備用模型
                elif model:
                    ai_prob = model.predict_proba(X)[0][1]

            except Exception:
                ai_prob = 0.5

            # ===== fallback 強化（關鍵升級）=====
            # 當AI信心太低 → 用規則模型補強
            if abs(ai_prob - 0.5) < 0.05:

                rule_score = 0

                # 趨勢
                if htf == 1:
                    rule_score += 0.25
                else:
                    rule_score -= 0.25

                # 動能
                if mid_trend == 1:
                    rule_score += 0.15
                else:
                    rule_score -= 0.15

                # breakout
                if breakout == 1:
                    rule_score += 0.25
                elif breakout == -1:
                    rule_score -= 0.25

                # macro
                rule_score += macro_bias * 0.1

                # triangle
                if triangle == 1:
                    rule_score += 0.05

                ai_prob = 0.5 + rule_score

            # clamp
            ai_prob = max(0.05, min(ai_prob, 0.95))

            # ===== AI主導進場（v4 最終盈利版）=====

            # ===== 動態平滑（避免卡0.5）=====
            if last_signal is not None:
                score = 0.75 * ai_prob + 0.25 * last_signal
            else:
                score = ai_prob

            # ===== 自適應噪音（低信心才探索）=====
            if abs(score - 0.5) < 0.08:
                noise = np.random.uniform(-0.05, 0.05)
            else:
                noise = np.random.uniform(-0.02, 0.02)
            score = max(0.05, min(score + noise, 0.95))

            # ===== 小週期主導（Dual Flow v2）=====
            confluence = 0

            # 小週期優先（動能）
            if mid_trend == 1:
                confluence += 1
            elif mid_trend == -1:
                confluence += 1

            # breakout 直接加權（不再綁定HTF）
            if breakout != 0:
                confluence += 1

            # 大週期只做「方向加權」（不再卡死）
            if htf == mid_trend:
                confluence += 1
            else:
                # 不一致 → 視為回調（仍給權重）
                confluence += 0.5

            score += confluence * 0.08

            # ===== Volume bias =====
            if volume_spike:
                if buy_pressure:
                    score += 0.06
                elif sell_pressure:
                    score -= 0.06

            # 吸籌/出貨 → 反轉警告
            if absorption:
                score *= 0.9

            # ===== Sweep 反轉加權 =====
            if sweep_high:
                score -= 0.12
            if sweep_low:
                score += 0.12

            # ===== Regime強化（趨勢才重倉）=====
            if regime == "bull_trend_strong":
                score += 0.25
            elif regime == "bull_trend":
                score += 0.12
            elif regime == "bear_trend_strong":
                score -= 0.25
            elif regime == "bear_trend":
                score -= 0.12

            # ===== Range市場幾乎不做 =====
            if regime == "range":
                score = 0.5 + (score - 0.5) * 0.4

            # ===== Macro強化（避免逆勢）=====
            if macro_bias == 1 and score > 0.5:
                score += 0.08
            elif macro_bias == -1 and score < 0.5:
                score -= 0.08
            else:
                score *= 0.95

            # ===== 新聞影響強化（增加時事判斷權重）=====
            if news_text:
                analysis = analyze_news_text(news_text)
                news_bias = analysis["bias"]  # -2 到 2
                news_score_adjust = news_bias * 0.08  # 將新聞 bias 轉換為 score 調整（-0.16 到 0.16）
                score += news_score_adjust
                score = max(0.05, min(score, 0.95))  # 確保在範圍內

            # ===== 最終決策（含進場 / TP / SL / 倉位）=====
            entry = price
            atr = df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1]

            # 預設
            final = "觀望"
            sl = None
            tp = None
            position_size = 0

            # ===== 提前進場機制（升級）=====
            early_entry = False
            if abs(score - 0.5) > 0.18 and htf == mid_trend:
                early_entry = True

            # ===== 回調模式（Pullback Trading）=====
            pullback_long = False
            pullback_short = False

            # 強多 → 回調做空
            if regime in ["bull_trend", "bull_trend_strong"] and mid_trend == -1:
                if volume_spike or breakout == -1:
                    pullback_short = True

            # 強空 → 回調做多
            if regime in ["bear_trend", "bear_trend_strong"] and mid_trend == 1:
                if volume_spike or breakout == 1:
                    pullback_long = True

            # ===== 假突破過濾（量價）=====
            fake_breakout = False
            if breakout != 0 and not volume_spike:
                fake_breakout = True
            if absorption or sweep_high or sweep_low:
                fake_breakout = True

            # ===== 相對性過濾（BTC）=====
            if breakout == 1 and btc_change < 0:
                fake_breakout = True
            if breakout == -1 and btc_change > 0:
                fake_breakout = True

            # 放寬條件，解決高分卻觀望問題
            if regime != "range" and abs(score - 0.5) > (0.05 - event_risk*0.03):

                # ===== 低信心過濾（避免亂單） =====
                if abs(score - 0.5) < 0.12:
                    final = "觀望（低信心）"

                # ===== 三角模式 v2（提前進場 + 突破加碼） =====
                if triangle == 1:

                    upper = df_15m["high"].tail(20).max()
                    lower = df_15m["low"].tail(20).min()
                    range_size = upper - lower

                    # 靠近上緣 → 做空
                    if price > upper - range_size * 0.2 and breakout == 0:
                        final = "🔺 三角上緣做空"

                        sl = upper
                        risk = sl - entry
                        tp = entry - risk * 1.8

                    # 靠近下緣 → 做多
                    elif price < lower + range_size * 0.2 and breakout == 0:
                        final = "🔻 三角下緣做多"

                        sl = lower
                        risk = entry - sl
                        tp = entry + risk * 1.8

                    # 突破 → 強訊號（加碼）
                    elif breakout == 1:
                        final = "🚀 三角突破做多"

                        sl = lower
                        risk = entry - sl
                        tp = entry + risk * 2.5

                    elif breakout == -1:
                        final = "🚀 三角跌破做空"

                        sl = upper
                        risk = sl - entry
                        tp = entry - risk * 2.5

                else:

                    # 對稱多空門檻：調整 AI 信號邏輯，提升做空觸發機率
                    if score > 0.52:
                        final = "🚀 做多"

                        recent_low_15 = df_15m["low"].tail(10).min()
                        sl = recent_low_15

                        risk = entry - sl
                        rr = 2.2 if regime.endswith("strong") else 1.6
                        tp = entry + risk * rr

                    elif score < 0.48:
                        final = "🚀 做空"

                        recent_high_15 = df_15m["high"].tail(10).max()
                        sl = recent_high_15

                        risk = sl - entry
                        rr = 2.2 if regime.endswith("strong") else 1.6
                        tp = entry - risk * rr

                    # 反彈單：把回調旗標接入實際開單（偏保守門檻）
                    elif pullback_long and score >= 0.45:
                        final = "↩️ 反彈做多"
                        recent_low_pb = min(df_5m["low"].tail(6).min(), df_15m["low"].tail(6).min())
                        sl = recent_low_pb
                        risk = max(entry - sl, atr * 0.45, entry * 0.001)
                        tp = entry + risk * 1.5

                    elif pullback_short and score <= 0.55:
                        final = "↩️ 反彈做空"
                        recent_high_pb = max(df_5m["high"].tail(6).max(), df_15m["high"].tail(6).max())
                        sl = recent_high_pb
                        risk = max(sl - entry, atr * 0.45, entry * 0.001)
                        tp = entry - risk * 1.5

                    # ===== 倉位（動態） =====
                    confidence = abs(score - 0.5) * 2

                    if regime in ["bull_trend_strong", "bear_trend_strong"]:
                        base = 0.5
                    elif regime in ["bull_trend", "bear_trend"]:
                        base = 0.35
                    else:
                        base = 0.2

                    if confidence > 0.7:
                        position_size = base
                    elif confidence > 0.5:
                        position_size = base * 0.7
                    elif confidence > 0.3:
                        position_size = base * 0.5
                    else:
                        position_size = base * 0.3

                    if losing_streak >= MAX_LOSS_STREAK:
                        position_size *= 0.5

                    # 三角策略調整（提前單較小，突破加碼）
                    if "三角" in final:
                        if "突破" in final:
                            position_size *= 1.2
                        else:
                            position_size *= 0.7

            # ===== 修正長短單 TP/SL（方向 + 最小風險距離） =====
            if final != "觀望":
                final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

            # ===== 中文時事解讀 =====
            macro_text = "中性"
            if event_risk >= 2:
                macro_text += "｜⚠️重大事件"
            elif event_risk == 1:
                macro_text += "｜⚠️波動增加"
            if macro_bias > 0:
                macro_text = "偏多（美股↑ / 美元↓ / 新聞利多）" + macro_text[2:] if macro_text.startswith("中性") else macro_text
            elif macro_bias < 0:
                macro_text = "偏空（美股↓ / 美元↑ / 新聞利空）" + macro_text[2:] if macro_text.startswith("中性") else macro_text

            # ===== AI判斷依據說明（技術來源強化版）=====
            reason = []

            # HTF 趨勢（MA25）
            if htf == 1:
                reason.append("多頭趨勢（4H MA25 上方）")
            else:
                reason.append("空頭趨勢（4H MA25 下方）")

            # MID 動能（MACD）
            if mid_trend == 1:
                reason.append("動能偏多（30m MACD > Signal）")
            else:
                reason.append("動能偏空（30m MACD < Signal）")

            # Breakout（結構突破）
            if breakout == 1:
                reason.append("突破高點（5m 結構突破）")
            elif breakout == -1:
                reason.append("跌破低點（5m 結構跌破）")

            # Regime（市場結構）
            if regime == "bull_trend_strong":
                reason.append("強多趨勢（1H+4H 同向 + 高波動）")
            elif regime == "bear_trend_strong":
                reason.append("強空趨勢（1H+4H 同向 + 高波動）")
            elif regime == "range":
                reason.append("盤整（趨勢不一致）")

            # Triangle（三角收斂）
            if triangle == 1:
                reason.append("三角收斂（高低點收斂）")

            # FVG（流動性缺口）
            if fvg_low and fvg_high:
                reason.append(f"FVG缺口（{fvg_low:.0f}-{fvg_high:.0f}）")

            # Macro（時事）
            if macro_bias > 0:
                reason.append("宏觀偏多（美股↑ / DXY↓）")
            elif macro_bias < 0:
                reason.append("宏觀偏空（美股↓ / DXY↑）")

            # News（時事）
            if news_bias > 0:
                reason.append("市場利多新聞（ETF/採用/上漲）")
            elif news_bias < 0:
                reason.append("市場利空新聞（監管/駭客/拋售）")

            # Event risk（重大波動）
            if event_risk >= 2:
                reason.append("重大事件（高波動風險）")
            elif event_risk == 1:
                reason.append("事件風險（波動提升）")

            # AI輸出
            reason.append(f"AI概率（{ai_prob:.2f}）")

            reason_text = " | ".join(reason)

            # ===== 市場狀態中文轉換 =====
            regime_map = {
                "bull_trend_strong": "強多趨勢",
                "bull_trend": "多頭趨勢",
                "range": "盤整",
                "bear_trend_strong": "強空趨勢",
                "bear_trend": "空頭趨勢"
            }

            regime_text = regime_map.get(regime, regime)

            # ===== 統一輸出訊號格式 =====
            if "做多" in final:
                display_signal = "🚀 做多"
            elif "做空" in final:
                display_signal = "🚀 做空"
            else:
                display_signal = final

            # ===== 訊息格式（進場優先顯示）=====
            msg = ""

            if final != "觀望":
                msg += f"📍 進場: {entry:.2f}\n"

                if tp is not None:
                    msg += f"🎯 止盈: {tp:.2f}\n"
                else:
                    msg += "🎯 止盈: N/A\n"

                if sl is not None:
                    msg += f"🛑 止損: {sl:.2f}\n"
                else:
                    msg += "🛑 止損: N/A\n"

                msg += f"💰 倉位: {int(position_size*100)}%\n\n"
            
            # 提取訊息中的進場/止盈/止損價格（確保與網址一致）
            entry_display_str = f"{entry:.2f}" if final != "觀望" else None
            tp_display_str = f"{tp:.2f}" if (final != "觀望" and tp is not None) else "0.0"
            sl_display_str = f"{sl:.2f}" if (final != "觀望" and sl is not None) else "0.0"

            msg += (
                f"🤖 AI信號：{display_signal}\n"
                f"📊 信心值: {ai_prob:.2f}\n"
                f"📈 勝率: {(performance['win']/performance['total'] if performance['total']>0 else 0):.2%}\n"
                f"🌍 市場狀態: {regime_text}\n"
                f"📰 時事判斷: {macro_text}\n"
                f"{news_text}"
                f"🧠 判斷依據: {reason_text}"
            )
            # Fix spam log（觀望不要一直print）
            if final != "觀望":
                print(msg)

            # ===== 強制進場（解決AI高信心但被過濾掉） =====
            if final == "觀望":
                if score > 0.7:
                    final = "🚀 做多（強制）"
                    recent_low_15 = df_15m["low"].tail(10).min()
                    sl = recent_low_15
                    risk = entry - sl
                    tp = entry + risk * 2

                elif score < 0.3:
                    final = "🚀 做空（強制）"
                    recent_high_15 = df_15m["high"].tail(10).max()
                    sl = recent_high_15
                    risk = sl - entry
                    tp = entry - risk * 2

            # 強制單也必須再次經過自動修正，避免繞過前面的保護
            if final != "觀望":
                final, sl, tp = auto_fix_trade_plan(final, entry, sl, tp, atr)

            # ===== 開單頻率 + 訊號去重（核心修正）=====
            now_ts = time.time()

            TRADE_COOLDOWN = 300  # 拉長避免過度交易

            # 先做去重與冷卻判斷，再決定是否跳過

            # ===== 同方向去重（用方向，不用字串） =====
            current_direction = get_signal_direction(final)
            last_direction_simple = get_signal_direction(last_trade_signal) if last_trade_signal else None

            # ===== 防洗單 v6 =====
            if current_direction == last_direction_simple:
                # 價格變動太小 → 不開單
                if last_entry_price is not None:
                    price_change = abs(price - last_entry_price) / price
                    if price_change < MIN_PRICE_CHANGE:
                        final = "觀望（防洗單-價格過近）"

                # 信號變化太小 → 不開單
                if last_signal is not None:
                    if abs(score - last_signal) < MIN_SIGNAL_DIFF:
                        final = "觀望（防洗單-信號重複）"

            if final != "觀望":
                # ===== 冷卻防洗單 =====
                if now_ts - last_trade_time < TRADE_COOLDOWN:
                    final = "觀望（冷卻中）"

                # ===== 價格變動過小 =====
                elif last_entry_price is not None:
                    price_change = abs(price - last_entry_price) / price
                    if price_change < MIN_PRICE_CHANGE:
                        final = "觀望（價格未達門檻）"

            # ===== 最終過濾 =====
            if final.startswith("觀望"):
                continue

            # ===== 最終安全檢查：拒絕假突破低信心單 =====
            if fake_breakout and abs(score - 0.5) < 0.22:
                continue

            if final != "觀望":

                # 保險：再次確認沒有持倉
                if active_trade["open"]:
                    continue

                # 防止同一訊號重複刷
                if last_signal_cache == msg:
                    continue

                print("📤 發送 Telegram")
                send_telegram(msg, priority=True, pin=True)
                last_signal_cache = msg
                last_trade_time = now_ts
                last_trade_signal = final
                last_entry_price = price
                last_direction = final

                # ===== 建立真實交易 =====
                direction = "long" if "做多" in final else "short"

                active_trade["direction"] = direction
                active_trade["entry"] = float(entry)
                active_trade["avg_entry"] = float(entry)
                active_trade["tp"] = tp
                active_trade["sl"] = sl
                base_size = _safe_float(position_size, 0.0)
                if base_size <= 0:
                    base_size = 0.2
                active_trade["size"] = float(min(1 / 3, max(base_size, 0.1)))
                active_trade["max_size"] = 1 / 3
                active_trade["min_size"] = max(0.1, active_trade["size"] * 0.3)
                active_trade["add_count"] = 0
                active_trade["reduce_count"] = 0
                active_trade["last_adjust_ts"] = 0.0
                active_trade["open_ts"] = time.time()
                active_trade["tp_decay_count"] = 0
                active_trade["open"] = True

                # ===== 實際向 Binance 下市價開倉單 + 掛 TP/SL =====
                if BINANCE_API_KEY and BINANCE_API_SECRET:
                    open_side = "BUY" if direction == "long" else "SELL"
                    order_qty = active_trade["size"]
                    lev_ok = binance_set_leverage("ETHUSDT", BINANCE_LEVERAGE)
                    if not lev_ok:
                        send_telegram(
                            f"⚠️ Binance 槓桿設定失敗，已使用帳戶現有槓桿設定繼續開倉",
                            priority=True,
                        )
                    order_ok = binance_futures_market_order(
                        "ETHUSDT", open_side, order_qty, reduce_only=False
                    )
                    if order_ok:
                        send_telegram(
                            f"✅ Binance 已自動開單 | 方向: {direction} | "
                            f"數量: {order_qty:.3f} ETH | 槓桿: {BINANCE_LEVERAGE}x | orderId: {order_ok}",
                            priority=True,
                        )
                        if tp is not None and sl is not None:
                            tp_sl_ok = binance_place_tp_sl_orders(
                                "ETHUSDT", direction, float(tp), float(sl),
                                quantity=order_qty,
                            )
                            if not tp_sl_ok:
                                send_telegram(
                                    "⚠️ TP/SL 掛單部分失敗，請手動確認交易所掛單狀態",
                                    priority=True,
                                )
                    else:
                        send_telegram(
                            f"⚠️ Binance 開倉失敗，僅記錄虛擬倉位（{direction}）",
                            priority=True,
                        )

                send_position_keyboard(
                    direction,
                    float(entry),
                    tp,
                    sl,
                    active_trade["size"],
                    entry_display=entry_display_str,
                    tp_display=tp_display_str,
                    sl_display=sl_display_str,
                )

            # ===== 記錄（未來價格）=====
            future_price = price
            time.sleep(1.2)

            new_price = WS_PRICE if WS_PRICE else price

            # （原簡易績效追蹤區塊已移除，現由真實交易管理統計）

            # ===== 更乾淨學習（避免噪音）=====
            # AI 標籤（根據實際進場方向與結果）
            if last_direction and last_entry_price:
                if "做多" in last_direction:
                    label = 1 if new_price > last_entry_price else 0
                elif "做空" in last_direction:
                    label = 1 if new_price < last_entry_price else 0
                else:
                    continue
            else:
                continue

            log_data(features, label)

            # ===== 即時學習（核心升級）=====
            update_online_model(features, label)

            # 每60秒嘗試訓練一次（更穩定）
            if int(time.time()) % 60 == 0:
                train_model()

            # ===== 更新信號（平滑 + 防洗單記錄）=====
            last_signal = score
            # ===== 每日報告 =====
            if time.time() - last_report_time > 3600:  # 每1小時
                winrate = performance["win"] / performance["total"] if performance["total"] > 0 else 0

                report = (
                    f"📊 交易報告\n"
                    f"總交易: {performance['total']}\n"
                    f"勝率: {winrate:.2%}\n"
                    f"勝: {performance['win']} / 敗: {performance['loss']}"
                )

                send_telegram(report)
                last_report_time = time.time()

            time.sleep(0.8)

        except Exception as e:
            print("error:", repr(e))
            time.sleep(3)

# =============================
# start
# =============================
if __name__ == "__main__":
    print("🔥 AI 接管版啟動")
    run_bot()
