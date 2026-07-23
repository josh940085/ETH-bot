"""News aggregation, classification, panel formatting, and Discord delivery."""
import datetime, html, json, os, pickle, re, threading, time, warnings
from collections import deque
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import numpy as np
import requests
from n8n_client import post_n8n_notification
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from runtime_config import is_truthy as _is_truthy
from runtime_paths import data_path, ensure_parent_dir

def _safe_float(value, default=0.0):
    try: return float(value)
    except Exception: return float(default)

def _safe_int(value, default=0):
    try: return int(value)
    except Exception: return int(default)

def _write_json_atomic(path, payload):
    try:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception as exc: print(f"⚠️ 寫入 {path.name} 失敗: {exc}")

def _write_pickle_atomic(path, payload):
    target = Path(path)
    try:
        ensure_parent_dir(target); tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, target)
    except Exception as exc: print(f"⚠️ 寫入 {target.name} 失敗: {exc}")

def _read_json_file(path, default):
    try:
        target = Path(path)
        if not target.exists(): return default
        with target.open("r", encoding="utf-8") as handle: data = json.load(handle)
        return data if isinstance(data, type(default)) else default
    except Exception: return default

def _write_json_file(path, payload):
    _write_json_atomic(path, payload)

HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
NEWS_CACHE = {"news": 0, "event": 0, "news_list": [], "ts": 0}
NEWS_MODEL_PATH = data_path("news_model.pkl")
NEWS_VECTORIZER_PATH = data_path("news_vectorizer.pkl")
NEWS_MODEL_META_PATH = data_path("news_model_meta.json")
NEWS_PERFORMANCE_LOG = data_path("news_predictions.jsonl")
NEWS_LEARNING_BUFFER = data_path("learning_buffer.pkl")
NEWS_EVAL_PENDING_PATH = data_path("news_eval_pending.pkl")
NEWS_STATS_CACHE_PATH = data_path("news_stats_cache.json")
NEWS_PUSH_DEDUPE_PATH = data_path("news_push_dedupe.json")
NEWS_MODEL_VERSION = 3
news_model = news_vectorizer = NEWS_EVAL_PENDING = None
PREDICTION_ACCURACY_CACHE = {"cache_key": None, "stats": None}
INCREMENTAL_LEARNING_ENABLED = True
MIN_PREDICTIONS_FOR_RETRAIN = 50
NEWS_EVAL_HORIZON_SEC, NEWS_EVAL_MAX_OVERDUE_SEC = 1800, 3600
NEWS_EVAL_MIN_MOVE_RATE, NEWS_EVAL_STRONG_MOVE_RATE = 0.0012, 0.0035
NEWS_EVAL_QUEUE_MAX, NEWS_EVAL_PROCESS_INTERVAL_SEC, NEWS_RETRAIN_MIN_INTERVAL_SEC = 400, 15.0, 900.0
TRANSLATION_CACHE, _CURRENT_MARKET_PRICE = {}, 0.0
DISCORD_WEBHOOK, DISCORD_NEWS = os.getenv("DISCORD_WEBHOOK", ""), os.getenv("DISCORD_NEWS", "")
DISCORD_AUTO_DELETE_HOURS = max(0.0, _safe_float(os.getenv("DISCORD_AUTO_DELETE_HOURS", 24.0), 24.0))
DISCORD_AUTO_DELETE_SEC = int(DISCORD_AUTO_DELETE_HOURS * 3600)

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


def _prepare_news_text_for_model(text):
    """將新聞文字轉成更適合模型學習的穩定格式。"""
    prepared = normalize_news_text(text)
    prepared = re.sub(r"https?://\S+", " ", prepared)
    prepared = prepared.replace("&amp;", " and ")
    prepared = re.sub(r"[$#]([A-Za-z]{2,12})", r" \1 ", prepared)
    prepared = re.sub(r"[^\w\u4e00-\u9fff%./:+-]+", " ", prepared)
    prepared = prepared.lower()
    prepared = re.sub(r"\s+", " ", prepared).strip()
    return prepared


def _news_has_term(text: str, term: str) -> bool:
    """Match news terms on word boundaries so short symbols such as ETH do not
    accidentally match unrelated words such as whether or method."""
    haystack = str(text or "").lower()
    needle = str(term or "").lower().strip()
    if not needle:
        return False
    if re.search(r"[\u4e00-\u9fff]", needle):
        # Chinese financial terms are commonly joined directly to numbers or
        # neighboring characters (for example, 暴跌3% or 台股重挫).
        return needle in haystack
    pattern = r"(?<![a-z0-9])" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _news_has_any(text: str, terms) -> bool:
    return any(_news_has_term(text, term) for term in terms)


GLOBAL_EQUITY_INDEX_TERMS = [
    # Global / North America / Latin America
    "s&p 500", "nasdaq", "dow jones", "russell 2000", "tsx", "s&p/tsx",
    "ipc mexico", "bmv ipc", "ibovespa", "bovespa", "merval", "ipsa", "colcap",
    # Europe
    "stoxx 600", "euro stoxx", "ftse 100", "ftse", "dax", "cac 40", "ibex 35",
    "ftse mib", "aex", "smi", "omx", "obx", "psi 20", "atx", "bel 20",
    "wig20", "bux", "px index", "moex", "bist 100", "athex", "iseq",
    # Asia Pacific
    "nikkei", "topix", "hang seng", "hsi", "csi 300", "shanghai composite",
    "shenzhen component", "taiex", "twse", "kospi", "kosdaq", "nifty 50",
    "sensex", "sti", "set index", "idx composite", "jakarta composite", "jci",
    "klci", "fbm klci", "psei", "vn-index", "vn index", "asx 200", "nzx 50",
    "kse 100", "dsex", "cse all share",
    # Middle East / Africa
    "tadawul", "tasi", "ta-35", "egx 30", "jse all share", "ftse/jse",
    "nse all share", "ngx all share", "masi", "dfm general", "adx general", "qe index",
    # Common Chinese market/index names
    "美股", "加拿大股市", "墨西哥股市", "巴西股市", "阿根廷股市", "英股", "德股",
    "法股", "義大利股市", "西班牙股市", "瑞士股市", "荷蘭股市", "歐股", "日股",
    "韓股", "港股", "陸股", "中國股市", "a股", "印度股市", "新加坡股市", "泰股",
    "印尼股市", "馬股", "菲股", "越股", "澳股", "紐西蘭股市", "台股", "臺股",
    "台灣股市", "臺灣股市", "加權指數", "櫃買指數", "沙烏地股市", "以色列股市",
    "南非股市", "埃及股市", "奈及利亞股市",
]

GLOBAL_EQUITY_COUNTRY_TERMS = [
    "united states", "u.s.", "us", "canada", "mexico", "brazil", "argentina", "chile",
    "colombia", "peru", "united kingdom", "uk", "britain", "germany", "france", "italy",
    "spain", "portugal", "netherlands", "belgium", "switzerland", "austria", "ireland",
    "sweden", "norway", "denmark", "finland", "poland", "czech republic", "hungary",
    "greece", "romania", "russia", "turkey", "japan", "china", "hong kong", "taiwan",
    "south korea", "korea", "india", "singapore", "thailand", "indonesia", "malaysia",
    "philippines", "vietnam", "australia", "new zealand", "pakistan", "bangladesh",
    "sri lanka", "saudi arabia", "israel", "united arab emirates", "uae", "qatar",
    "kuwait", "egypt", "south africa", "nigeria", "morocco", "kenya",
]


def _is_global_equity_market_scope(text: str) -> bool:
    """Recognize national stock markets and the main indices across regions."""
    low = normalize_news_text(text).lower()
    if _news_has_any(low, GLOBAL_EQUITY_INDEX_TERMS):
        return True

    market_noun = r"(?:stock\s+markets?|stocks?|shares?|equities|equity\s+markets?|indices|index)"
    for country in GLOBAL_EQUITY_COUNTRY_TERMS:
        country_pattern = re.escape(country).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![a-z0-9]){country_pattern}(?![a-z0-9]).{{0,24}}\b{market_noun}\b", low):
            return True
        if re.search(rf"\b{market_noun}\b.{{0,24}}(?<![a-z0-9]){country_pattern}(?![a-z0-9])", low):
            return True
    return False


def _news_relevance_reason(text: str) -> str:
    """Return a global-financial-market relevance group, or empty for noise."""
    low = normalize_news_text(text).lower()
    if not low:
        return ""

    low_value_commentary = _news_has_any(low, [
        "live levels", "price prediction", "technical analysis", "chart of the day",
        "why is bitcoin up", "why is bitcoin down", "why is ethereum up",
        "why is ethereum down", "what happened to bitcoin", "what happened to ethereum",
    ])
    if low_value_commentary:
        return ""

    major_company = _news_has_any(low, [
        "nvidia", "apple", "microsoft", "amazon", "alphabet", "google", "meta platforms",
        "tesla", "tsmc", "taiwan semiconductor", "broadcom", "samsung electronics",
        "jpmorgan", "goldman sachs", "blackrock", "berkshire hathaway",
    ])
    major_company_catalyst = _news_has_any(low, [
        "earnings", "revenue", "profit", "guidance", "forecast", "sales warning",
        "misses estimates", "beats estimates", "chip export", "export ban", "antitrust",
        "investigation", "outage", "default", "bankruptcy", "acquisition", "takeover",
    ])
    if major_company and major_company_catalyst:
        return "mega_cap"

    corporate_noise = _news_has_any(low, [
        "ipo", "price target", "per share", "insider sale", "insider sells", "form 4",
        "form 144", "class a shares", "appoints", "named ceo",
    ])
    personal_or_case_noise = _news_has_any(low, [
        "sentenced to prison", "sentences", "prison for", "advisor gets", "employee jailed",
    ])
    if corporate_noise or personal_or_case_noise:
        return ""

    direct_crypto = _news_has_any(low, [
        "bitcoin", "btc", "ethereum", "eth", "ether", "crypto", "cryptocurrency",
        "cryptocurrencies", "digital asset", "digital assets", "stablecoin", "usdt",
        "usdc", "defi", "staking", "blockchain", "web3", "spot etf", "crypto etf",
        "bitcoin etf", "ethereum etf", "on-chain", "hash rate", "halving", "altcoin",
        "binance", "coinbase", "kraken", "bybit", "okx", "deribit", "grayscale",
        "microstrategy", "exchange hack", "smart contract", "layer 2", "layer2",
    ])
    if direct_crypto:
        return "crypto"

    macro_release = _news_has_any(low, [
        "cpi", "pce", "inflation", "consumer prices", "producer prices", "nonfarm",
        "non-farm", "payrolls", "jobless claims", "jobs report", "unemployment",
        "gdp", "retail sales", "industrial production", "pmi", "consumer confidence",
        "business confidence", "trade balance", "current account", "wage growth",
    ])
    systemic_risk = _news_has_any(low, [
        "bank collapse", "bank run", "banking crisis", "credit crisis", "liquidity crisis",
        "financial crisis", "sovereign default", "us default", "u.s. default", "recession",
        "stagflation", "debt ceiling", "credit downgrade", "emergency bailout",
    ])
    if macro_release or systemic_risk:
        return "macro"

    central_bank = _news_has_any(low, [
        "fed", "federal reserve", "fomc", "powell", "ecb", "european central bank",
        "bank of england", "boe", "bank of japan", "boj", "people's bank of china",
        "pboc", "swiss national bank", "snb", "bank of canada", "reserve bank of australia",
        "rba",
    ]) and _news_has_any(low, [
        "interest rate", "rates", "rate hike", "rate cut", "inflation", "liquidity",
        "balance sheet", "bond", "yield", "currency", "policy", "economy", "stimulus",
        "quantitative easing", "quantitative tightening", "intervention",
    ])
    if central_bank:
        return "central_bank"

    market_move = _news_has_any(low, [
        "rise", "rises", "rally", "higher", "gain", "gains", "fall", "falls", "slump",
        "drop", "drops", "plunge", "plunges", "lower", "selloff", "sell-off", "surge",
        "surges", "jump", "jumps", "slide", "slides", "tumble", "tumbles", "rebound",
        "rebounds", "advance", "advances",
        "record high", "record low", "one-month low", "one-year high", "volatile",
        "上漲", "大漲", "暴漲", "勁揚", "重挫", "大跌", "暴跌", "崩跌", "下跌",
        "跌逾", "跌破", "摜破", "賣壓", "拋售",
    ])
    global_equities = _news_has_any(low, [
        "s&p 500", "nasdaq", "dow jones", "wall st", "wall street", "stock futures",
        "stock market", "global stocks", "world stocks", "european shares", "asian shares",
        "msci world", "emerging markets", "risk-off", "risk off",
    ]) or _is_global_equity_market_scope(low)
    if global_equities and market_move:
        return "global_equities"

    rates_fx = _news_has_any(low, [
        "treasury yields", "bond yields", "government bonds", "treasuries", "gilts", "bunds",
        "dxy", "dollar index", "us dollar", "u.s. dollar", "euro", "yen", "yuan", "renminbi",
        "pound sterling", "swiss franc", "foreign exchange", "forex",
    ]) and (market_move or _news_has_any(low, ["intervention", "rate", "yield", "inflation"]))
    if rates_fx:
        return "rates_fx"

    commodity_move = _news_has_any(low, [
        "oil", "crude", "brent", "wti", "gold", "silver", "copper", "natural gas",
        "iron ore", "commodity", "commodities",
    ]) and (market_move or _news_has_any(low, [
        "supply", "disruption", "sanction", "inventory", "opec", "inflation", "tariff",
    ]))
    if commodity_move:
        return "commodities"

    trade_policy = _news_has_any(low, [
        "tariff", "tariffs", "trade war", "export control", "export controls", "sanction",
        "sanctions", "capital controls",
    ]) and _news_has_any(low, [
        "united states", "u.s.", "us", "china", "european union", "japan", "russia", "india",
        "canada", "mexico", "global", "world",
    ])
    if trade_policy:
        return "trade_policy"

    oil_shock = _news_has_any(low, ["oil", "crude", "brent", "hormuz", "energy crisis"]) and _news_has_any(low, [
        "iran", "israel", "war", "strike", "sanction", "supply", "disruption", "crisis",
        "surge", "rise", "rises", "jump", "drop", "falls", "inflation",
    ])
    geopolitical_shock = (
        _news_has_any(low, ["iran", "israel", "hormuz", "middle east"]) and
        _news_has_any(low, ["war", "strike", "missile", "attack", "ceasefire", "blockade", "sanction"])
    ) or (
        _news_has_any(low, ["russia", "ukraine"]) and
        _news_has_any(low, ["ceasefire", "invasion", "escalation", "nato", "sanction", "nuclear weapon"])
    ) or (
        _news_has_any(low, ["china", "taiwan", "taiwan strait"]) and
        _news_has_any(low, ["blockade", "military drill", "invasion", "missile", "sanction", "export control"])
    )
    if oil_shock or geopolitical_shock:
        return "geopolitical"

    return ""


def _major_equity_market_move_override(text: str):
    """Return a deterministic bias for an unusually large national-index move.

    The generic news model is primarily trained on English crypto/macro text and
    can otherwise label local-language market crash headlines as neutral.
    """
    low = normalize_news_text(text).lower()
    if not _is_global_equity_market_scope(low):
        return 0, 0.0

    percent_moves = [
        _safe_float(value, 0.0)
        for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", low)
    ]
    point_moves = [
        _safe_float(value.replace(",", ""), 0.0)
        for value in re.findall(r"(?:跌|挫|漲|升|摜|plunge|drop|fall|rise|jump)[^\d]{0,12}([\d,]+(?:\.\d+)?)\s*(?:點|points?)", low)
    ]
    taiwan_market = _news_has_any(low, [
        "taiex", "twse", "taiwan stocks", "taiwan shares", "taiwan weighted",
        "台股", "臺股", "台灣股市", "臺灣股市", "加權指數", "櫃買指數",
    ])
    large_magnitude = max(percent_moves or [0.0]) >= 2.0 or (
        taiwan_market and max(point_moves or [0.0]) >= 800.0
    )
    strong_bear = _news_has_any(low, [
        "重挫", "大跌", "暴跌", "崩跌", "跌逾", "摜破", "plunge", "plunges",
        "plunged", "tumble", "tumbles", "slump", "slumps", "crash", "selloff", "sell-off",
    ])
    strong_bull = _news_has_any(low, [
        "大漲", "暴漲", "勁揚", "漲逾", "surge", "surges", "soar", "soars",
        "jump", "jumps", "rally", "rallies",
    ])
    if strong_bear or (large_magnitude and _news_has_any(low, ["跌", "挫", "摜", "drop", "fall", "lower"])):
        return -2, 0.82
    if strong_bull or (large_magnitude and _news_has_any(low, ["漲", "升", "rise", "gain", "higher"])):
        return 2, 0.82
    return 0, 0.0


def _news_dedupe_key(text: str) -> str:
    """Create a source-independent key and collapse small headline suffix changes."""
    key = _prepare_news_text_for_model(re.sub(r"^\[[^\]]+\]\s*", "", str(text or "")))
    key = re.sub(r"\s+(?:on|at)\s+(?:the\s+)?(?:nyse|nasdaq|wall street)$", "", key)
    return key[:220]


def _news_dedupe_tokens(text: str):
    prepared = _news_dedupe_key(text)
    stop_words = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
        "the", "to", "with", "after", "before", "amid", "says", "said", "update", "live",
    }
    return {
        token for token in re.findall(r"[a-z0-9%]+|[\u4e00-\u9fff]{2,}", prepared)
        if len(token) >= 2 and token not in stop_words
    }


def _news_titles_are_similar(first: str, second: str) -> bool:
    first_key = _news_dedupe_key(first)
    second_key = _news_dedupe_key(second)
    if not first_key or not second_key:
        return False
    if first_key == second_key:
        return True
    first_tokens = _news_dedupe_tokens(first_key)
    second_tokens = _news_dedupe_tokens(second_key)
    intersection = len(first_tokens & second_tokens)
    if intersection < 4:
        return False
    union = len(first_tokens | second_tokens)
    smaller = min(len(first_tokens), len(second_tokens))
    jaccard = intersection / max(1, union)
    containment = intersection / max(1, smaller)
    threshold = max(0.6, min(0.95, _safe_float(os.getenv("NEWS_PUSH_SIMILARITY_THRESHOLD", 0.78), 0.78)))
    return jaccard >= threshold or (containment >= 0.9 and abs(len(first_tokens) - len(second_tokens)) <= 4)


def _register_news_push_if_new(text: str, now_ts=None) -> bool:
    """Persist recently pushed headlines and reject source/wording variants."""
    now_ts = _safe_float(now_ts, time.time())
    window_sec = max(300.0, _safe_float(os.getenv("NEWS_PUSH_DEDUPE_WINDOW_SEC", 43200), 43200))
    history = getattr(_register_news_push_if_new, "_history", None)
    if not isinstance(history, list):
        payload = _read_json_file(NEWS_PUSH_DEDUPE_PATH, {})
        history = payload.get("items", []) if isinstance(payload, dict) else []
    fresh = []
    for item in history:
        if not isinstance(item, dict):
            continue
        item_ts = _safe_float(item.get("ts"), 0.0)
        item_text = str(item.get("text") or item.get("key") or "").strip()
        if item_text and now_ts - item_ts <= window_sec:
            fresh.append({"key": str(item.get("key") or _news_dedupe_key(item_text)), "text": item_text, "ts": item_ts})

    candidate_key = _news_dedupe_key(text)
    if not candidate_key:
        return False
    for item in fresh:
        if candidate_key == item.get("key") or _news_titles_are_similar(text, item.get("text", "")):
            _register_news_push_if_new._history = fresh
            return False

    fresh.append({"key": candidate_key, "text": normalize_news_text(text)[:300], "ts": now_ts})
    fresh = fresh[-500:]
    _register_news_push_if_new._history = fresh
    _write_json_file(NEWS_PUSH_DEDUPE_PATH, {"updated_at": now_ts, "window_sec": window_sec, "items": fresh})
    return True


def _is_market_relevant_news(text: str) -> bool:
    """判斷新聞是否足以影響全球股票、債券、外匯、商品或加密市場。"""
    return bool(_news_relevance_reason(text))




def _sanitize_news_label(label):
    try:
        return max(-2, min(2, int(label)))
    except Exception:
        return None

def _load_learning_buffer_samples(max_per_label=40):
    try:
        with open(NEWS_LEARNING_BUFFER, "rb") as f:
            raw_buffer = pickle.load(f)
    except Exception:
        return []

    if not isinstance(raw_buffer, list):
        return []

    unique_samples = []
    seen_texts = set()

    for item in reversed(raw_buffer):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        text, label = item
        prepared = _prepare_news_text_for_model(text)
        clean_label = _sanitize_news_label(label)
        if (
            not prepared
            or clean_label is None
            or prepared in seen_texts
            or not _is_market_relevant_news(prepared)
        ):
            continue

        seen_texts.add(prepared)
        unique_samples.append((prepared, clean_label))

    unique_samples.reverse()

    grouped = {}
    for text, label in unique_samples:
        grouped.setdefault(label, []).append((text, label))

    selected = []
    for label in sorted(grouped):
        selected.extend(grouped[label][-max_per_label:])

    return selected


def _build_news_vectorizer():
    return FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    stop_words="english",
                    lowercase=False,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    max_features=2000,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )


def _build_news_model():
    return VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)),
            ("sgd", SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced", random_state=42)),
            ("nb", ComplementNB(alpha=0.35)),
        ],
        voting="soft",
    )

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
    if news_model is not None and news_vectorizer is not None:
        return

    sample_map = {}
    for text, label in NEWS_TRAINING_DATA:
        prepared = _prepare_news_text_for_model(text)
        clean_label = _sanitize_news_label(label)
        if prepared and clean_label is not None:
            sample_map[prepared] = clean_label

    for text, label in _load_learning_buffer_samples(max_per_label=40):
        sample_map[text] = label

    if not sample_map:
        sample_map = {
            _prepare_news_text_for_model(text): label
            for text, label in NEWS_TRAINING_DATA
            if _prepare_news_text_for_model(text)
        }

    texts = list(sample_map.keys())
    labels = [sample_map[text] for text in texts]

    news_vectorizer = _build_news_vectorizer()
    X = news_vectorizer.fit_transform(texts)
    y = np.array(labels)

    news_model = _build_news_model()
    news_model.fit(X, y)

    try:
        ensure_parent_dir(NEWS_MODEL_PATH)
        ensure_parent_dir(NEWS_VECTORIZER_PATH)
        _write_pickle_atomic(NEWS_MODEL_PATH, news_model)
        _write_pickle_atomic(NEWS_VECTORIZER_PATH, news_vectorizer)
        _write_json_file(
            NEWS_MODEL_META_PATH,
            {
                "version": NEWS_MODEL_VERSION,
                "sample_count": len(texts),
                "trained_at": datetime.datetime.now().isoformat(),
            },
        )
    except Exception:
        pass


def load_news_model(force_retrain=False):
    global news_model, news_vectorizer
    if news_model is not None and news_vectorizer is not None and not force_retrain:
        return

    meta = _read_json_file(NEWS_MODEL_META_PATH, {})
    needs_retrain = force_retrain or meta.get("version") != NEWS_MODEL_VERSION

    if not needs_retrain:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with open(NEWS_MODEL_PATH, "rb") as f:
                    news_model = pickle.load(f)
                with open(NEWS_VECTORIZER_PATH, "rb") as f:
                    news_vectorizer = pickle.load(f)

            if not hasattr(news_model, "predict_proba") or not hasattr(news_vectorizer, "transform"):
                raise ValueError("news model/vectorizer is incompatible")
            return
        except Exception as e:
            print(f"♻️ 新聞模型已改用當前環境重建: {e}")

    news_model = None
    news_vectorizer = None
    train_news_model()



def predict_news_sentiment_with_confidence(text):
    """預測新聞情緒 + 置信度分數（新函數，更智能）"""
    global news_model, news_vectorizer
    if news_model is None:
        load_news_model()
    if news_model is None:
        return 0, 0.33  # 預設中性，低置信度

    try:
        prepared = _prepare_news_text_for_model(text)
        if not prepared:
            return 0, 0.33

        X = news_vectorizer.transform([prepared])
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
def log_prediction_result(
    news_text,
    predicted_bias,
    actual_market_move=None,
    correct=None,
    actual_bias=None,
    ai_confidence=None,
    source="News",
    schedule_eval=True,
):
    """記錄預測結果用於增量學習和精準度評估"""
    try:
        # 只記錄可能影響全球金融市場的新聞，避免無關資訊污染訓練資料
        if not _is_market_relevant_news(news_text):
            return

        prepared = _prepare_news_text_for_model(news_text)
        raw_text = normalize_news_text(news_text)
        if not prepared or not raw_text or "\n" in raw_text:
            return

        recent = getattr(log_prediction_result, "_recent", {})
        now_ts = time.time()
        dedupe_window_sec = 900 if actual_market_move is None and correct is None else 0
        if dedupe_window_sec > 0 and now_ts - recent.get(prepared, 0) < dedupe_window_sec:
            return

        recent[prepared] = now_ts
        if len(recent) > 2000:
            recent = {k: v for k, v in recent.items() if now_ts - v < 3600}
        log_prediction_result._recent = recent

        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "news": raw_text[:150],
            "news_key": prepared[:120],
            "predicted_bias": predicted_bias,
            "actual_move": actual_market_move,
            "actual_bias": actual_bias,
            "is_correct": correct,
            "ai_confidence": ai_confidence,
            "source": str(source or "News")[:60],
        }

        ensure_parent_dir(NEWS_PERFORMANCE_LOG)
        with open(NEWS_PERFORMANCE_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        PREDICTION_ACCURACY_CACHE["cache_key"] = None
        PREDICTION_ACCURACY_CACHE["stats"] = None

        if schedule_eval and actual_market_move is None and correct is None:
            _queue_news_prediction_for_evaluation(
                news_text=raw_text,
                predicted_bias=predicted_bias,
                ai_confidence=ai_confidence,
                source=source,
            )
    except Exception:
        pass


def get_prediction_accuracy():
    """計算模型預測準確度"""
    default_stats = {"accuracy": 0, "total": 0, "correct": 0}
    try:
        log_path = Path(NEWS_PERFORMANCE_LOG)
        if not log_path.exists():
            return default_stats

        stat = log_path.stat()
        cache_key = f"{int(stat.st_mtime)}:{stat.st_size}"

        if PREDICTION_ACCURACY_CACHE.get("cache_key") == cache_key and PREDICTION_ACCURACY_CACHE.get("stats"):
            return dict(PREDICTION_ACCURACY_CACHE["stats"])

        cached = _read_json_file(NEWS_STATS_CACHE_PATH, {})
        if cached.get("cache_key") == cache_key:
            stats = {
                "accuracy": round(float(cached.get("accuracy", 0)), 2),
                "total": int(cached.get("total", 0)),
                "correct": int(cached.get("correct", 0)),
            }
            PREDICTION_ACCURACY_CACHE["cache_key"] = cache_key
            PREDICTION_ACCURACY_CACHE["stats"] = dict(stats)
            return stats

        total = 0
        correct = 0
        
        with open(log_path, "r", encoding="utf-8") as f:
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
        stats = {"accuracy": round(accuracy, 2), "total": total, "correct": correct}
        PREDICTION_ACCURACY_CACHE["cache_key"] = cache_key
        PREDICTION_ACCURACY_CACHE["stats"] = dict(stats)
        _write_json_file(NEWS_STATS_CACHE_PATH, {"cache_key": cache_key, **stats})
        return stats
    except Exception:
        return default_stats


def update_learning_buffer(news_text, true_label):
    """將新樣本添加到增量學習緩衝區"""
    try:
        # 只學習可能影響全球金融市場的新聞
        if not _is_market_relevant_news(news_text):
            return

        prepared = _prepare_news_text_for_model(news_text)
        clean_label = _sanitize_news_label(true_label)
        if not prepared or clean_label is None:
            return

        buffer = []
        try:
            with open(NEWS_LEARNING_BUFFER, "rb") as f:
                buffer = pickle.load(f)
        except Exception:
            buffer = []

        if not isinstance(buffer, list):
            buffer = []

        buffer.append((prepared, clean_label))

        deduped = []
        seen = set()
        for text, label in reversed(buffer):
            key = (str(text), _sanitize_news_label(label))
            if not key[0] or key[1] is None or key in seen:
                continue
            seen.add(key)
            deduped.append((key[0], key[1]))

        deduped.reverse()

        # 緩衝區最多保留 200 個樣本
        if len(deduped) > 200:
            deduped = deduped[-200:]

        _write_pickle_atomic(NEWS_LEARNING_BUFFER, deduped)
    except Exception:
        pass


def incremental_train_news_model():
    """增量學習：結合原始訓練數據 + 學習緩衝區新樣本進行重新訓練"""
    global news_model, news_vectorizer
    news_model = None
    news_vectorizer = None

    try:
        train_news_model()
        meta = _read_json_file(NEWS_MODEL_META_PATH, {})
        print(f"✓ 增量學習完成：使用 {meta.get('sample_count', len(NEWS_TRAINING_DATA))} 個樣本重新訓練模型")
    except Exception as e:
        print(f"✗ 增量學習失敗: {e}")


# 新聞情緒/事件分析（更穩定的分類）
def analyze_news_text(raw_text, log_result=True):
    """更穩定的新聞分類：拆分多空 / 事件 / 影響，避免單一關鍵字誤判。"""
    raw_text = str(raw_text or "").strip()
    text = _prepare_news_text_for_model(raw_text)
    if not text:
        return {
            "sentiment": "中性",
            "impact": "影響有限",
            "bias": 0,
            "event_risk": 0,
            "score": 0,
            "ai_bias": 0,
            "ai_confidence": 0.33,
            "tags": ["empty_text"],
            "is_event": False,
            "fusion_method": "empty_text",
        }

    # ===== 直接使用 AI 模型判斷，不再依賴關鍵字規則 =====
    ai_bias, ai_confidence = predict_news_sentiment_with_confidence(text)
    tags = [f"ai_conf:{ai_confidence:.2f}"]
    fusion_note = "ai_only"
    final_bias = _refine_neutral_bias(text, ai_bias, ai_confidence)
    event_risk = 0

    equity_move_bias, equity_move_confidence = _major_equity_market_move_override(raw_text)
    if equity_move_bias:
        final_bias = equity_move_bias
        ai_confidence = max(ai_confidence, equity_move_confidence)
        tags.append("major_global_equity_market_move")
        fusion_note = "major_global_equity_market_move_override"
        event_risk = 1

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

    if log_result:
        log_prediction_result(raw_text, final_bias, ai_confidence=ai_confidence)

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
def _news_direction_icon(analysis):
    bias = _safe_int((analysis or {}).get("bias"), 0) if isinstance(analysis, dict) else 0
    if bias > 0:
        return "🔴"
    if bias < 0:
        return "🟢"
    return "🟡"


def build_news_message(news_text, now_time=None, analysis=None):
    if now_time is None:
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

    source_match = re.match(r"^\[([^\]]+)\]\s*", str(news_text))
    source = source_match.group(1).strip() if source_match else "News"
    raw_text = re.sub(r"^\[[^\]]+\]\s*", "", str(news_text)).strip()
    zh_text = translate_news_to_zh(raw_text)

    # ===== AI 交易解讀 =====
    analysis = analysis if isinstance(analysis, dict) else analyze_news_text(raw_text)
    sentiment = analysis["sentiment"]
    impact = analysis["impact"]
    confidence = analysis["ai_confidence"]
    direction_icon = _news_direction_icon(analysis)
    
    # ===== 顯示 AI 學習狀態 =====
    accuracy_info = get_prediction_accuracy()
    accuracy_str = f"準率: {accuracy_info['accuracy']}% ({accuracy_info['correct']}/{accuracy_info['total']})" if accuracy_info['total'] > 0 else "準率: 初始化中"

    return (
        f"{direction_icon} 新聞(中文): {zh_text}\n"
        f"⏰ {now_time}\n"
        f"━━━━━━━━━━━━━━\n"
        f"市場快訊（即時）\n"
        f"來源: {source}\n"
        f"📊 解讀: {sentiment}\n"
        f"🎯 置信度: {confidence:.1%}\n"
        f"🧠 {accuracy_str}\n"
        f"🔥 影響: {impact}\n"
        f"📝 原文: {raw_text}\n"
        f"━━━━━━━━━━━━━━"
    )


NEWS_LOCATION_HINTS = [
    (("canada", "air canada", "加拿大"), "加拿大", 56.1304, -106.3468),
    (("united states", "u.s.", " us ", "美國", "華爾街", "wall street", "new york", "紐約"), "美國", 39.8283, -98.5795),
    (("washington", "white house", "federal reserve", "fed", "白宮", "聯準會"), "華盛頓", 38.9072, -77.0369),
    (("china", "beijing", "中國", "北京"), "中國", 35.8617, 104.1954),
    (("taiwan", "台灣", "臺灣"), "台灣", 23.6978, 120.9605),
    (("japan", "tokyo", "日本", "東京"), "日本", 36.2048, 138.2529),
    (("south korea", "korea", "seoul", "韓國", "首爾"), "韓國", 35.9078, 127.7669),
    (("india", "new delhi", "印度"), "印度", 20.5937, 78.9629),
    (("pakistan", "巴基斯坦"), "巴基斯坦", 30.3753, 69.3451),
    (("iran", "tehran", "伊朗"), "伊朗", 32.4279, 53.6880),
    (("israel", "gaza", "jerusalem", "以色列", "加薩", "耶路撒冷"), "以色列/加薩", 31.0461, 34.8516),
    (("syria", "damascus", "敘利亞", "大馬士革"), "敘利亞", 34.8021, 38.9968),
    (("russia", "moscow", "俄羅斯", "莫斯科"), "俄羅斯", 61.5240, 105.3188),
    (("ukraine", "kyiv", "烏克蘭", "基輔"), "烏克蘭", 48.3794, 31.1656),
    (("united kingdom", "britain", "london", "英國", "倫敦"), "英國", 55.3781, -3.4360),
    (("france", "paris", "法國", "巴黎"), "法國", 46.2276, 2.2137),
    (("germany", "berlin", "德國", "柏林"), "德國", 51.1657, 10.4515),
    (("europe", "eurozone", "eu ", "歐洲", "歐盟", "歐元區"), "歐洲", 54.5260, 15.2551),
    (("zambia", "尚比亞", "贊比亞"), "尚比亞", -13.1339, 27.8493),
    (("south africa", "南非"), "南非", -30.5595, 22.9375),
    (("brazil", "巴西"), "巴西", -14.2350, -51.9253),
    (("mexico", "墨西哥"), "墨西哥", 23.6345, -102.5528),
    (("australia", "澳洲", "澳大利亞"), "澳洲", -25.2744, 133.7751),
    (("saudi", "riyadh", "沙烏地", "沙特"), "沙烏地阿拉伯", 23.8859, 45.0792),
]


def infer_news_location(title, title_zh="", source=""):
    haystack = f" {title or ''} {title_zh or ''} {source or ''} ".lower()
    for keys, name, lat, lon in NEWS_LOCATION_HINTS:
        if any(str(key).lower() in haystack for key in keys):
            return {"location": name, "lat": lat, "lon": lon}
    return {}


def build_panel_news_items(news_list, limit=5):
    items = []
    seen = set()

    for raw_item in list(news_list or [])[: max(limit * 2, limit)]:
        raw_text = str(raw_item or "").strip()
        if not raw_text:
            continue

        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", raw_text)
        if match:
            source = match.group(1).strip() or "News"
            title = match.group(2).strip()
        else:
            source = "News"
            title = raw_text

        if not title:
            continue
        if not _is_market_relevant_news(title):
            continue

        key = _news_dedupe_key(title)
        if key in seen:
            continue
        seen.add(key)

        analysis = analyze_news_text(title, log_result=False)
        bias = _safe_int(analysis.get("bias"), 0) if isinstance(analysis, dict) else 0
        confidence = _safe_float(analysis.get("confidence"), 0.0) if isinstance(analysis, dict) else 0.0
        title_zh = translate_news_to_zh(title)
        item = {
            "source": source[:40],
            "title": title[:220],
            "title_zh": str(title_zh or title)[:220],
            "bias": bias,
            "confidence": round(confidence, 4),
            "ts": int(time.time()),
        }
        item.update(infer_news_location(title, title_zh, source))
        items.append(item)

        if len(items) >= limit:
            break

    return items


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
        "bank", "banks", "digital franc", "institutional", "taiex", "twse",
        "台股", "臺股", "台灣股市", "臺灣股市", "加權指數", "櫃買指數",
        "重挫", "大跌", "暴跌", "崩跌", "跌逾", "摜破"
    ]) or _is_global_equity_market_scope(text) or len(text.split()) >= 6


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
        # 0. 台灣市場（中央社與自由財經官方 RSS）
        ("https://feeds.feedburner.com/rsscna/finance", "中央社財經"),
        ("https://news.ltn.com.tw/rss/business.xml", "自由財經"),

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

        # 4. 外匯分析
        ("https://www.investing.com/rss/forex.rss", "Technical Analysis"),
    ]
    if _is_truthy(os.getenv("RSS_ENABLE_FOREXLIVE", "0")):
        feeds.append(("https://www.forexlive.com/feed/", "ForexLive"))

    source_batches = []
    for feed_url, source_name in feeds:
        now_feed = time.time()
        cooldown_key = f"rss_cooldown_until_{source_name.lower()}"
        cooldown_until = _safe_float(getattr(fetch_macro_rss_news, cooldown_key, 0.0), 0.0)
        if cooldown_until > now_feed:
            continue
        try:
            source_batches.append(fetch_rss_news(feed_url, source_name))
            setattr(fetch_macro_rss_news, f"rss_fail_count_{source_name.lower()}", 0)
        except Exception as e:
            now_err = now_feed
            key = f"rss_err_{source_name.lower()}"
            fail_key = f"rss_fail_count_{source_name.lower()}"
            fail_count = _safe_int(getattr(fetch_macro_rss_news, fail_key, 0), 0) + 1
            setattr(fetch_macro_rss_news, fail_key, fail_count)
            if fail_count >= max(2, _safe_int(os.getenv("RSS_SOURCE_COOLDOWN_FAILS", 3), 3)):
                cooldown_sec = max(300, _safe_int(os.getenv("RSS_SOURCE_FAIL_COOLDOWN_SEC", 3600), 3600))
                setattr(fetch_macro_rss_news, cooldown_key, now_err + cooldown_sec)
            last_err = getattr(fetch_macro_rss_news, key, 0)
            if now_err - last_err > max(300, _safe_int(os.getenv("RSS_ERROR_LOG_INTERVAL_SEC", 1800), 1800)):
                print(f"⚠️ {source_name} RSS 暫時略過，已進入冷卻:", repr(e))
                setattr(fetch_macro_rss_news, key, now_err)

    # Round-robin the sources so the first high-volume feed cannot consume the
    # global 50-item cap and starve Taiwan or crypto-specific feeds.
    aggregated = []
    max_batch_len = max((len(batch) for batch in source_batches), default=0)
    for item_index in range(max_batch_len):
        for batch in source_batches:
            if item_index < len(batch):
                aggregated.append(batch[item_index])

    dedup = []
    seen = set()
    for item in aggregated:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "RSS")).strip() or "RSS"
        text = normalize_news_text(item.get("text", ""))
        if not text:
            continue
        key = _news_dedupe_key(text)
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
            if not text or not _is_market_relevant_news(text):
                continue
            key = _news_dedupe_key(text)
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
                refresh_rss_news_cache.seen_news.add(_news_dedupe_key(text))
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

            for item in normalized_items[8:]:
                refresh_rss_news_cache.seen_news.add(_news_dedupe_key(item["text"]))

            refresh_rss_news_cache.bootstrapped_news = True
        else:
            for item in normalized_items:
                src = item["source"]
                text = item["text"]
                seen_key = _news_dedupe_key(text)
                if seen_key in refresh_rss_news_cache.seen_news:
                    continue

                refresh_rss_news_cache.seen_news.add(seen_key)
                analysis = analyze_news_text(text)
                news_bias += int(analysis.get("bias", 0))
                event_risk += int(analysis.get("event_risk", 0))
                news_list.append(f"[{src}] {text[:200]}")

        # 若本輪沒有新快訊，回退為近期標題（僅保留市場相關），避免監控面板長期顯示「暫無資料」
        if not news_list and latest_news:
            news_list = [n for n in latest_news if _is_market_relevant_news(n)]

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

def _sanitize_pending_news_eval_item(item):
    if not isinstance(item, dict):
        return None

    raw_news = normalize_news_text(item.get("news", item.get("text", "")))
    prepared = _prepare_news_text_for_model(raw_news)
    if not raw_news or not prepared or not _is_market_relevant_news(raw_news):
        return None

    predicted_bias = _sanitize_news_label(item.get("predicted_bias"))
    if predicted_bias is None:
        return None

    source = str(item.get("source", "News")).strip() or "News"
    entry_price = _safe_float(item.get("entry_price"), 0.0)
    entry_ts = _safe_float(item.get("entry_ts"), 0.0)
    due_ts = _safe_float(item.get("due_ts"), 0.0)
    if entry_price <= 0 or entry_ts <= 0:
        return None
    if due_ts <= entry_ts:
        due_ts = entry_ts + NEWS_EVAL_HORIZON_SEC

    raw_key = str(item.get("news_key", "")).strip().lower()
    news_key = raw_key or f"{source.lower()}|{prepared[:180]}"
    ai_confidence = max(0.0, min(1.0, _safe_float(item.get("ai_confidence"), 0.0)))

    return {
        "news_key": news_key[:220],
        "source": source[:60],
        "news": raw_news[:240],
        "predicted_bias": predicted_bias,
        "ai_confidence": ai_confidence,
        "entry_price": entry_price,
        "entry_ts": entry_ts,
        "due_ts": due_ts,
    }


def _load_pending_news_eval_queue():
    global NEWS_EVAL_PENDING

    if isinstance(NEWS_EVAL_PENDING, list):
        return NEWS_EVAL_PENDING

    try:
        with open(NEWS_EVAL_PENDING_PATH, "rb") as f:
            raw_queue = pickle.load(f)
    except Exception:
        raw_queue = []

    if not isinstance(raw_queue, list):
        raw_queue = []

    queue = []
    seen = set()
    for item in raw_queue:
        clean = _sanitize_pending_news_eval_item(item)
        if clean is None:
            continue
        dedupe_key = (clean["news_key"], int(clean["entry_ts"]))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        queue.append(clean)

    NEWS_EVAL_PENDING = queue[-NEWS_EVAL_QUEUE_MAX:]
    if len(NEWS_EVAL_PENDING) != len(raw_queue):
        _save_pending_news_eval_queue(NEWS_EVAL_PENDING)
    return NEWS_EVAL_PENDING


def _save_pending_news_eval_queue(queue=None):
    global NEWS_EVAL_PENDING

    if queue is None:
        queue = NEWS_EVAL_PENDING if isinstance(NEWS_EVAL_PENDING, list) else []

    clean_queue = []
    for item in queue:
        clean = _sanitize_pending_news_eval_item(item)
        if clean is not None:
            clean_queue.append(clean)

    NEWS_EVAL_PENDING = clean_queue[-NEWS_EVAL_QUEUE_MAX:]
    try:
        _write_pickle_atomic(NEWS_EVAL_PENDING_PATH, NEWS_EVAL_PENDING)
    except Exception as e:
        print(f"⚠️ 儲存新聞待驗證隊列失敗: {e}")


def _queue_news_prediction_for_evaluation(news_text, predicted_bias, ai_confidence=None, source="News"):
    if not INCREMENTAL_LEARNING_ENABLED:
        return False

    raw_news = normalize_news_text(news_text)
    prepared = _prepare_news_text_for_model(raw_news)
    clean_bias = _sanitize_news_label(predicted_bias)
    entry_price = _safe_float(_CURRENT_MARKET_PRICE, 0.0)
    now_ts = time.time()
    if not raw_news or not prepared or clean_bias is None or entry_price <= 0:
        return False

    queue = _load_pending_news_eval_queue()
    news_key = f"{str(source or 'News').strip().lower()}|{prepared[:180]}"

    for item in reversed(queue):
        if item.get("news_key") != news_key:
            continue
        if now_ts - _safe_float(item.get("entry_ts"), 0.0) < NEWS_EVAL_HORIZON_SEC:
            return False
        break

    queue.append(
        {
            "news_key": news_key,
            "source": str(source or "News"),
            "news": raw_news,
            "predicted_bias": clean_bias,
            "ai_confidence": max(0.0, min(1.0, _safe_float(ai_confidence, 0.0))),
            "entry_price": entry_price,
            "entry_ts": now_ts,
            "due_ts": now_ts + NEWS_EVAL_HORIZON_SEC,
        }
    )
    _save_pending_news_eval_queue(queue)
    return True


def _classify_news_market_move(move_rate):
    move = _safe_float(move_rate, 0.0)
    abs_move = abs(move)
    if abs_move < NEWS_EVAL_MIN_MOVE_RATE:
        return 0
    if move >= NEWS_EVAL_STRONG_MOVE_RATE:
        return 2
    if move >= NEWS_EVAL_MIN_MOVE_RATE:
        return 1
    if move <= -NEWS_EVAL_STRONG_MOVE_RATE:
        return -2
    return -1


def _is_news_prediction_correct(predicted_bias, actual_bias):
    predicted = _sanitize_news_label(predicted_bias)
    actual = _sanitize_news_label(actual_bias)
    if predicted is None or actual is None:
        return None
    if predicted == 0 or actual == 0:
        return predicted == actual
    return (predicted > 0) == (actual > 0)


def _maybe_retrain_news_model(labeled_total=None):
    if not INCREMENTAL_LEARNING_ENABLED:
        return False

    total = max(0, _safe_int(labeled_total, 0))
    bucket = total // max(1, MIN_PREDICTIONS_FOR_RETRAIN)
    if bucket <= 0:
        return False

    state = getattr(_maybe_retrain_news_model, "_state", {"last_bucket": 0, "last_run_ts": 0.0})
    now_ts = time.time()
    if bucket <= _safe_int(state.get("last_bucket"), 0):
        return False
    if (now_ts - _safe_float(state.get("last_run_ts"), 0.0)) < NEWS_RETRAIN_MIN_INTERVAL_SEC:
        return False

    incremental_train_news_model()
    _maybe_retrain_news_model._state = {
        "last_bucket": bucket,
        "last_run_ts": now_ts,
    }
    return True


def _process_pending_news_evaluations(current_price):
    global _CURRENT_MARKET_PRICE
    now_ts = time.time()
    current = _safe_float(current_price, 0.0)
    if current > 0:
        _CURRENT_MARKET_PRICE = current
    last_run_ts = _safe_float(getattr(_process_pending_news_evaluations, "_last_run_ts", 0.0), 0.0)
    if (now_ts - last_run_ts) < NEWS_EVAL_PROCESS_INTERVAL_SEC:
        return 0
    _process_pending_news_evaluations._last_run_ts = now_ts

    price_ref = _safe_float(current_price, 0.0) or _safe_float(_CURRENT_MARKET_PRICE, 0.0)
    if price_ref <= 0:
        return 0

    queue = _load_pending_news_eval_queue()
    if not queue:
        return 0

    keep = []
    evaluated = 0
    stale = 0
    for item in queue:
        due_ts = _safe_float(item.get("due_ts"), 0.0)
        if due_ts > now_ts:
            keep.append(item)
            continue

        overdue_sec = now_ts - due_ts
        if overdue_sec > NEWS_EVAL_MAX_OVERDUE_SEC:
            stale += 1
            continue

        entry_price = _safe_float(item.get("entry_price"), 0.0)
        if entry_price <= 0:
            continue

        actual_move = (price_ref - entry_price) / max(entry_price, 1e-9)
        actual_bias = _classify_news_market_move(actual_move)
        predicted_bias = _sanitize_news_label(item.get("predicted_bias"))
        correct = _is_news_prediction_correct(predicted_bias, actual_bias)

        log_prediction_result(
            item.get("news", ""),
            predicted_bias,
            actual_market_move=round(actual_move, 6),
            correct=correct,
            actual_bias=actual_bias,
            ai_confidence=item.get("ai_confidence"),
            source=item.get("source", "News"),
            schedule_eval=False,
        )
        update_learning_buffer(item.get("news", ""), actual_bias)
        evaluated += 1

    if len(keep) != len(queue):
        _save_pending_news_eval_queue(keep)

    if stale > 0:
        print(f"🧹 已清除 {stale} 筆過期新聞驗證樣本")

    if evaluated > 0:
        stats = get_prediction_accuracy()
        retrained = _maybe_retrain_news_model(stats.get("total", 0))
        print(
            f"🧠 新聞驗證完成: {evaluated} 筆 | 準率 {stats.get('accuracy', 0)}% "
            f"({stats.get('correct', 0)}/{stats.get('total', 0)})"
            + (" | 已重訓新聞模型" if retrained else "")
        )

    return evaluated


def _discord_webhook_base_url(webhook_url: str) -> str:
    parsed = urlparse(str(webhook_url or "").strip())
    if not parsed.scheme or not parsed.netloc or not parsed.path:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _schedule_discord_message_delete(webhook_url: str, message_id: str, delay_sec: int):
    base_url = _discord_webhook_base_url(webhook_url)
    msg_id = str(message_id or "").strip()
    if not base_url or not msg_id or delay_sec <= 0:
        return

    def _delete_message():
        try:
            HTTP_SESSION.delete(f"{base_url}/messages/{msg_id}", timeout=8)
        except Exception as e:
            print("Discord auto-delete error:", e)

    timer = threading.Timer(delay_sec, _delete_message)
    timer.daemon = True
    timer.start()


def _post_discord_webhook(webhook_url: str, content: str, timeout: int = 5):
    url = str(webhook_url or "").strip()
    if not url:
        return

    payload = {"content": str(content or "")}
    destination = "discord_news" if url == str(DISCORD_NEWS or "").strip() else "discord_trade"
    res = post_n8n_notification(
        destination,
        payload,
        wait_for_response=DISCORD_AUTO_DELETE_SEC > 0,
        timeout=timeout,
        session=HTTP_SESSION,
    )

    if res is None:
        if DISCORD_AUTO_DELETE_SEC <= 0:
            HTTP_SESSION.post(url, json=payload, timeout=timeout)
            return

        # 需要 wait=true 才能拿到 message id，供後續刪除
        res = HTTP_SESSION.post(url, json=payload, params={"wait": "true"}, timeout=timeout)
        res.raise_for_status()
    elif DISCORD_AUTO_DELETE_SEC <= 0:
        return

    message_id = ""
    try:
        body = res.json() if res is not None else {}
        if isinstance(body, dict):
            message_id = str(body.get("id", "") or "")
    except Exception:
        message_id = ""

    if message_id:
        _schedule_discord_message_delete(url, message_id, DISCORD_AUTO_DELETE_SEC)
