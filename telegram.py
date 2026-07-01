import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

try:
	from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - urllib3 variant fallback
	NotOpenSSLWarning = None

if NotOpenSSLWarning is not None:
	warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import json
import os
import time

import requests

try:
	import fcntl
except ImportError:  # pragma: no cover - Windows fallback
	fcntl = None

from runtime_paths import REPO_DIR, ai_data_path, data_path, ensure_parent_dir


TELEGRAM_STATE_FILE = data_path(".telegram_state.json")
TELEGRAM_TOKEN = ""

TELEGRAM_POLL_BACKOFF_SEC = 1.0
TELEGRAM_POLL_BACKOFF_MAX = 60.0
TELEGRAM_POLL_TIMEOUT_BACKOFF_MAX = 12.0
TELEGRAM_GET_UPDATES_TIMEOUT_SEC = 8
TELEGRAM_HTTP_CONNECT_TIMEOUT_SEC = 4
TELEGRAM_HTTP_READ_TIMEOUT_SEC = TELEGRAM_GET_UPDATES_TIMEOUT_SEC + 4
TELEGRAM_POLL_LAST_ERROR_KEY = ""
TELEGRAM_POLL_LAST_LOG_TS = 0.0
TELEGRAM_HEALTH_EVENT_LIMIT = 400
TELEGRAM_HEALTH_RETENTION_SEC = 7 * 86400

HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "ETH-bot-telegram/1.0"})


def load_local_env():
	env_path = REPO_DIR / ".env"
	if not env_path.exists():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key:
			os.environ[key] = value


def configure_token(token=None):
	global TELEGRAM_TOKEN
	if token is None:
		TELEGRAM_TOKEN = str(os.getenv("TELEGRAM_TOKEN", "") or "").strip()
	else:
		TELEGRAM_TOKEN = str(token or "").strip()
	return TELEGRAM_TOKEN


def parse_telegram_state(raw: str) -> dict:
	try:
		payload = json.loads(raw) if str(raw).strip() else {}
		return payload if isinstance(payload, dict) else {}
	except Exception:
		return {}


def read_telegram_state_locked() -> dict:
	if not TELEGRAM_STATE_FILE.exists():
		return {}

	try:
		with TELEGRAM_STATE_FILE.open("r", encoding="utf-8") as fh:
			if fcntl is not None:
				fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
			raw = fh.read()
			if fcntl is not None:
				fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
		return parse_telegram_state(raw)
	except Exception:
		return {}


def update_telegram_state(mutator):
	try:
		TELEGRAM_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
		with TELEGRAM_STATE_FILE.open("a+", encoding="utf-8") as fh:
			if fcntl is not None:
				fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

			fh.seek(0)
			payload = parse_telegram_state(fh.read())
			result = mutator(payload)

			fh.seek(0)
			fh.truncate()
			fh.write(json.dumps(payload, ensure_ascii=False))
			fh.flush()
			os.fsync(fh.fileno())

			if fcntl is not None:
				fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
			return result
	except Exception as e:
		print(f"⚠️ 讀寫 telegram state 失敗: {e}")
		return None


def normalize_chat_id(value):
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	try:
		return str(int(text))
	except Exception:
		return text


def is_private_chat_id(chat_id) -> bool:
	normalized = normalize_chat_id(chat_id)
	if not normalized:
		return False
	try:
		return int(normalized) > 0
	except Exception:
		return False


def _remember_notification_chat_mutate(payload, chat_id):
	chat_text = normalize_chat_id(chat_id)
	if not chat_text:
		return

	try:
		chat_num = int(chat_text)
	except Exception:
		return

	if chat_num <= 0:
		return

	notify_ids = payload.get("notify_chat_ids")
	if not isinstance(notify_ids, list):
		notify_ids = []

	cleaned = []
	for item in notify_ids:
		item_text = normalize_chat_id(item)
		if not item_text:
			continue
		try:
			item_num = int(item_text)
		except Exception:
			continue
		if item_num > 0 and item_text not in cleaned:
			cleaned.append(item_text)

	if chat_text in cleaned:
		cleaned.remove(chat_text)
	cleaned.append(chat_text)
	payload["notify_chat_ids"] = cleaned[-5:]
	payload["last_private_chat_id"] = chat_num


def remember_notification_chat(chat_id):
	def _mutate(payload):
		_remember_notification_chat_mutate(payload, chat_id)

	update_telegram_state(_mutate)


def remove_notification_chat(chat_id):
	chat_text = normalize_chat_id(chat_id)
	if not chat_text:
		return

	def _mutate(payload):
		changed = False
		notify_ids = payload.get("notify_chat_ids")
		if isinstance(notify_ids, list):
			filtered = []
			for item in notify_ids:
				item_text = normalize_chat_id(item)
				if not item_text:
					continue
				if item_text == chat_text:
					changed = True
					continue
				filtered.append(item_text)
			payload["notify_chat_ids"] = filtered[-5:]

		last_private = normalize_chat_id(payload.get("last_private_chat_id"))
		if last_private and last_private == chat_text:
			payload["last_private_chat_id"] = ""
			changed = True

		return changed

	removed = update_telegram_state(_mutate)
	if removed:
		print(f"🧹 已移除失效 Telegram chat_id: {chat_text}")


def get_notification_chat_ids():
	targets = []
	seen = set()

	def _append(value):
		chat_text = normalize_chat_id(value)
		if not chat_text or chat_text in seen:
			return
		seen.add(chat_text)
		targets.append(chat_text)

	payload = read_telegram_state_locked()
	notify_ids = payload.get("notify_chat_ids")
	if isinstance(notify_ids, list):
		for item in notify_ids:
			try:
				if int(str(item).strip()) > 0:
					_append(item)
			except Exception:
				continue

	try:
		if int(str(payload.get("last_private_chat_id", "")).strip()) > 0:
			_append(payload.get("last_private_chat_id"))
	except Exception:
		pass

	for env_name in ("TELEGRAM_PRIVATE_CHAT_ID", "TELEGRAM_CHAT_ID"):
		try:
			if int(str(os.getenv(env_name, "")).strip()) > 0:
				_append(os.getenv(env_name))
		except Exception:
			pass

	return targets


def _truncate_telegram_health_text(value, limit=220):
	text = str(value or "").strip()
	if len(text) <= limit:
		return text
	return text[: max(0, limit - 3)].rstrip() + "..."


def _parse_telegram_error_payload(body):
	description = ""
	retry_after = 0
	try:
		payload = json.loads(str(body or "").strip())
		if isinstance(payload, dict):
			description = str(payload.get("description", "") or "").strip()
			parameters = payload.get("parameters")
			if isinstance(parameters, dict):
				try:
					retry_after = max(0, int(parameters.get("retry_after", 0) or 0))
				except Exception:
					retry_after = 0
	except Exception:
		description = ""

	if not description:
		description = _truncate_telegram_health_text(body)

	return {
		"description": description,
		"retry_after": retry_after,
	}


def inspect_telegram_delivery(status_code=None, body="", error=None):
	status_text = str(status_code or "").strip()
	try:
		status_num = int(status_text)
	except Exception:
		status_num = None

	payload = _parse_telegram_error_payload(body)
	description = str(payload.get("description", "") or "").strip()
	retry_after = int(payload.get("retry_after", 0) or 0)
	if error is not None and not description:
		description = _truncate_telegram_health_text(error)

	lower = description.lower()
	category = "unknown_error"
	ok = status_num == 200 and error is None

	if ok:
		category = "ok"
	elif status_num == 429 or "too many requests" in lower:
		category = "rate_limited"
	elif "bot was blocked by the user" in lower:
		category = "blocked_by_user"
	elif "user is deactivated" in lower:
		category = "user_deactivated"
	elif "chat not found" in lower:
		category = "chat_not_found"
	elif status_num == 401 or "unauthorized" in lower:
		category = "unauthorized"
	elif status_num == 403:
		category = "forbidden"
	elif status_num == 400:
		category = "bad_request"
	elif status_num is not None and status_num >= 500:
		category = "server_error"
	elif isinstance(error, requests.exceptions.Timeout) or "timeout" in str(error or "").lower():
		category = "timeout"
	elif error is not None:
		category = "network_error"

	return {
		"ok": ok,
		"status_code": status_num if status_num is not None else status_text,
		"category": category,
		"description": description,
		"retry_after": retry_after,
		"remove_chat": category in {"blocked_by_user", "user_deactivated", "chat_not_found"},
	}


def note_telegram_delivery_event(chat_id=None, ok=False, status_code=None, body="", error=None, context=""):
	info = inspect_telegram_delivery(status_code=status_code, body=body, error=error)
	now_ts = int(time.time())
	chat_text = normalize_chat_id(chat_id)
	event = {
		"ts": now_ts,
		"ok": bool(ok and info.get("ok")),
		"chat_id": chat_text or "",
		"status_code": info.get("status_code"),
		"category": str(info.get("category", "") or ""),
		"description": _truncate_telegram_health_text(info.get("description", "")),
		"retry_after": int(info.get("retry_after", 0) or 0),
		"context": str(context or "").strip(),
	}

	def _mutate(payload):
		events = payload.get("telegram_delivery_events")
		if not isinstance(events, list):
			events = []

		cutoff_ts = now_ts - TELEGRAM_HEALTH_RETENTION_SEC
		cleaned = []
		for item in events:
			if not isinstance(item, dict):
				continue
			try:
				item_ts = int(item.get("ts", 0) or 0)
			except Exception:
				continue
			if item_ts < cutoff_ts:
				continue
			cleaned.append(item)

		cleaned.append(event)
		payload["telegram_delivery_events"] = cleaned[-TELEGRAM_HEALTH_EVENT_LIMIT:]

		summary = payload.get("telegram_delivery_summary")
		if not isinstance(summary, dict):
			summary = {}
		summary["last_event_ts"] = now_ts
		if event["ok"]:
			summary["last_ok_ts"] = now_ts
		else:
			summary["last_error_ts"] = now_ts
			summary["last_error_category"] = event["category"]
			summary["last_error_description"] = event["description"]
			summary["last_error_chat_id"] = event["chat_id"]
		payload["telegram_delivery_summary"] = summary

	update_telegram_state(_mutate)
	return info


def get_follow_mode_enabled() -> bool:
	payload = read_telegram_state_locked()
	return bool(payload.get("follow_mode_enabled", False))


def set_follow_mode_enabled(value: bool):
	def _mutate(payload):
		payload["follow_mode_enabled"] = bool(value)

	update_telegram_state(_mutate)


def toggle_follow_mode_enabled() -> bool:
	def _mutate(payload):
		new_value = not bool(payload.get("follow_mode_enabled", False))
		payload["follow_mode_enabled"] = new_value
		return new_value

	return bool(update_telegram_state(_mutate))


def resolve_private_chat_id_for_controls(chat_id=None):
	candidate = normalize_chat_id(chat_id)
	if candidate:
		try:
			if int(candidate) > 0:
				return candidate
		except Exception:
			pass

	payload = read_telegram_state_locked()
	candidate = normalize_chat_id(payload.get("last_private_chat_id"))
	if candidate:
		try:
			if int(candidate) > 0:
				return candidate
		except Exception:
			pass

	return None


def _append_pending_command(chat_id, text, update_id, user_id=None, username="", first_name="", chat_type=""):
	def _mutate(payload):
		queue = payload.get("pending_commands")
		if not isinstance(queue, list):
			queue = []

		_remember_notification_chat_mutate(payload, chat_id)
		queue.append(
			{
				"chat_id": chat_id,
				"text": text,
				"user_id": user_id,
				"username": str(username or ""),
				"first_name": str(first_name or ""),
				"chat_type": str(chat_type or ""),
				"update_id": int(update_id),
				"ts": int(time.time()),
			}
		)
		payload["pending_commands"] = queue[-50:]
		payload["last_update_id"] = int(update_id)

	update_telegram_state(_mutate)


def _append_pending_callback(chat_id, callback_data, callback_id, message_id, update_id, user_id=None, username="", first_name="", chat_type=""):
	def _mutate(payload):
		queue = payload.get("pending_commands")
		if not isinstance(queue, list):
			queue = []

		_remember_notification_chat_mutate(payload, chat_id)
		queue.append(
			{
				"chat_id": chat_id,
				"text": f"__callback__:{callback_data}:{callback_id}:{message_id}",
				"user_id": user_id,
				"username": str(username or ""),
				"first_name": str(first_name or ""),
				"chat_type": str(chat_type or ""),
				"update_id": int(update_id),
				"ts": int(time.time()),
			}
		)
		payload["pending_commands"] = queue[-50:]
		payload["last_update_id"] = int(update_id)

	update_telegram_state(_mutate)


def _set_restart_requested(update_id):
	def _mutate(payload):
		payload["restart_requested"] = True
		payload["restart_requested_at"] = int(time.time())
		payload["last_update_id"] = int(update_id)

	update_telegram_state(_mutate)


def consume_restart_request() -> bool:
	def _mutate(payload):
		if not payload.get("restart_requested"):
			return False
		payload["restart_requested"] = False
		payload["restart_requested_at"] = int(time.time())
		return True

	return bool(update_telegram_state(_mutate))


def consume_supervisor_commands():
	def _mutate(payload):
		queue = payload.get("pending_commands")
		if not isinstance(queue, list) or not queue:
			return []
		items = list(queue)
		payload["pending_commands"] = []
		return items

	items = update_telegram_state(_mutate)
	return items if isinstance(items, list) else []


def send_message(chat_id, text, timeout=5, token=None):
	resolved_token = str(token or TELEGRAM_TOKEN or "").strip()
	if not resolved_token or chat_id is None:
		return

	try:
		response = HTTP_SESSION.post(
			f"https://api.telegram.org/bot{resolved_token}/sendMessage",
			data={"chat_id": chat_id, "text": text},
			timeout=timeout,
		)
		info = note_telegram_delivery_event(
			chat_id=chat_id,
			ok=response.status_code == 200,
			status_code=response.status_code,
			body=response.text,
			context="telegram.send_message",
		)
		if info.get("remove_chat"):
			remove_notification_chat(chat_id)
		return response
	except Exception as e:
		note_telegram_delivery_event(
			chat_id=chat_id,
			ok=False,
			status_code="no-response",
			error=e,
			context="telegram.send_message",
		)
		return None


def _is_telegram_poll_conflict_error(err) -> bool:
	text = str(err or "")
	return "409" in text and "Conflict" in text


def _is_telegram_poll_timeout_error(err) -> bool:
	if isinstance(err, requests.exceptions.Timeout):
		return True
	text = str(err or "")
	return "Read timed out" in text or "read timeout" in text.lower()


def _note_telegram_poll_success():
	global TELEGRAM_POLL_BACKOFF_SEC, TELEGRAM_POLL_LAST_ERROR_KEY, TELEGRAM_POLL_LAST_LOG_TS
	TELEGRAM_POLL_BACKOFF_SEC = 1.0
	TELEGRAM_POLL_LAST_ERROR_KEY = ""
	TELEGRAM_POLL_LAST_LOG_TS = 0.0


def _handle_telegram_poll_error(err):
	global TELEGRAM_POLL_BACKOFF_SEC, TELEGRAM_POLL_LAST_ERROR_KEY, TELEGRAM_POLL_LAST_LOG_TS

	now_ts = time.time()
	text = str(err or "")
	is_conflict = _is_telegram_poll_conflict_error(err)
	is_timeout = _is_telegram_poll_timeout_error(err)

	if is_conflict:
		backoff = max(15.0, TELEGRAM_POLL_BACKOFF_SEC)
		next_backoff = max(backoff * 2, 30.0)
		TELEGRAM_POLL_BACKOFF_SEC = min(TELEGRAM_POLL_BACKOFF_MAX, next_backoff)
	elif is_timeout:
		backoff = min(TELEGRAM_POLL_TIMEOUT_BACKOFF_MAX, max(1.0, TELEGRAM_POLL_BACKOFF_SEC))
		next_backoff = max(backoff * 1.6, 2.0)
		TELEGRAM_POLL_BACKOFF_SEC = min(TELEGRAM_POLL_TIMEOUT_BACKOFF_MAX, next_backoff)
	else:
		backoff = TELEGRAM_POLL_BACKOFF_SEC
		TELEGRAM_POLL_BACKOFF_SEC = min(TELEGRAM_POLL_BACKOFF_MAX, backoff * 2)

	if is_conflict:
		error_key = "telegram-409-conflict"
		message = f"⚠️ Telegram getUpdates 發生 409 Conflict，疑似另一個 bot 實例正在輪詢；{backoff:.0f}s 後重試"
		min_log_interval = max(backoff, 60.0)
	elif is_timeout:
		error_key = "telegram-read-timeout"
		message = f"ℹ️ Telegram 連線逾時（{backoff:.0f}s 後重試）: {text}"
		min_log_interval = max(backoff, 20.0)
	else:
		error_key = text
		message = f"⚠️ 讀取 Telegram 更新失敗（{backoff:.0f}s 後重試）: {text}"
		min_log_interval = max(backoff, 15.0)

	if error_key != TELEGRAM_POLL_LAST_ERROR_KEY or now_ts - TELEGRAM_POLL_LAST_LOG_TS >= min_log_interval:
		print(message)
		TELEGRAM_POLL_LAST_ERROR_KEY = error_key
		TELEGRAM_POLL_LAST_LOG_TS = now_ts

	time.sleep(backoff)


def poll_telegram_commands(token=None):
	resolved_token = str(token or TELEGRAM_TOKEN or "").strip()
	if not resolved_token:
		return

	payload = read_telegram_state_locked()
	last_update_id = payload.get("last_update_id")

	params = {"timeout": TELEGRAM_GET_UPDATES_TIMEOUT_SEC}
	if last_update_id is not None:
		try:
			params["offset"] = int(last_update_id) + 1
		except Exception:
			pass

	try:
		res = HTTP_SESSION.get(
			f"https://api.telegram.org/bot{resolved_token}/getUpdates",
			params=params,
			timeout=(TELEGRAM_HTTP_CONNECT_TIMEOUT_SEC, TELEGRAM_HTTP_READ_TIMEOUT_SEC),
		)
		res.raise_for_status()
		payload = res.json()
		updates = payload.get("result", []) if isinstance(payload, dict) else []
		_note_telegram_poll_success()
	except Exception as e:
		_handle_telegram_poll_error(e)
		updates = []

	for u in updates:
		update_id = u.get("update_id")
		msg = u.get("message", {})
		text = msg.get("text", "")
		chat_id = msg.get("chat", {}).get("id")
		chat_type = str(msg.get("chat", {}).get("type", "") or "")
		from_user = msg.get("from", {}) if isinstance(msg.get("from"), dict) else {}
		user_id = from_user.get("id")
		username = str(from_user.get("username", "") or "")
		first_name = str(from_user.get("first_name", "") or "")

		if update_id is None:
			continue

		cq = u.get("callback_query")
		if cq:
			cq_data = cq.get("data", "")
			cq_id = cq.get("id", "")
			cq_msg = cq.get("message", {})
			cq_msg_id = cq_msg.get("message_id")
			cq_chat_id = cq_msg.get("chat", {}).get("id")

			_append_pending_callback(
				cq_chat_id,
				cq_data,
				cq_id,
				cq_msg_id,
				update_id,
				user_id=cq.get("from", {}).get("id"),
				username=str(cq.get("from", {}).get("username", "") or ""),
				first_name=str(cq.get("from", {}).get("first_name", "") or ""),
				chat_type=str(cq_msg.get("chat", {}).get("type", "") or ""),
			)
			continue

		web_app_data = msg.get("web_app_data") if isinstance(msg, dict) else None
		if isinstance(web_app_data, dict):
			raw_web_app_data = str(web_app_data.get("data", "") or "").strip()
			action = ""
			if raw_web_app_data:
				try:
					parsed = json.loads(raw_web_app_data)
					if isinstance(parsed, dict):
						action = str(parsed.get("action", "") or "").strip()
				except Exception:
					action = raw_web_app_data

			if action:
				_append_pending_command(
					chat_id,
					f"__webapp__:{action}",
					update_id,
					user_id=user_id,
					username=username,
					first_name=first_name,
					chat_type=chat_type,
				)
				continue

		if not text:
			def _mutate(payload):
				_remember_notification_chat_mutate(payload, chat_id)
				payload["last_update_id"] = int(update_id)

			update_telegram_state(_mutate)
			continue

		if text.startswith("/restart"):
			_set_restart_requested(update_id)
			send_message(chat_id, "♻️ 已收到 /restart，將由啟動器同步並重啟。", token=resolved_token)
			continue

		_append_pending_command(
			chat_id,
			text,
			update_id,
			user_id=user_id,
			username=username,
			first_name=first_name,
			chat_type=chat_type,
		)


def fetch_telegram_commands(
	last_update_id=None,
	bot_supervisor=False,
	telegram_token=None,
	webapp_command_prefix="__webapp__:",
):
	resolved_token = str(telegram_token or TELEGRAM_TOKEN or "").strip()
	if not resolved_token:
		return [], last_update_id

	if bot_supervisor:
		commands = []
		pending_items = consume_supervisor_commands()
		newest_update_id = last_update_id

		for item in pending_items:
			if not isinstance(item, dict):
				continue

			update_id = item.get("update_id")
			try:
				if update_id is not None:
					update_id = int(update_id)
					newest_update_id = update_id if newest_update_id is None else max(int(newest_update_id), update_id)
			except Exception:
				update_id = None

			commands.append(
				{
					"update_id": update_id,
					"text": str(item.get("text", "") or ""),
					"chat_id": item.get("chat_id"),
					"user_id": item.get("user_id"),
					"username": str(item.get("username", "") or ""),
					"first_name": str(item.get("first_name", "") or ""),
					"chat_type": str(item.get("chat_type", "") or ""),
				}
			)

		return commands, newest_update_id

	params = {"timeout": TELEGRAM_GET_UPDATES_TIMEOUT_SEC}
	if last_update_id is not None:
		try:
			params["offset"] = int(last_update_id) + 1
		except Exception:
			pass

	try:
		res = HTTP_SESSION.get(
			f"https://api.telegram.org/bot{resolved_token}/getUpdates",
			params=params,
			timeout=(TELEGRAM_HTTP_CONNECT_TIMEOUT_SEC, TELEGRAM_HTTP_READ_TIMEOUT_SEC),
		)
		res.raise_for_status()
		updates = res.json().get("result", [])
		_note_telegram_poll_success()
	except Exception as e:
		_handle_telegram_poll_error(e)
		return [], last_update_id

	commands = []
	newest_update_id = last_update_id
	for u in updates:
		update_id = u.get("update_id")
		if update_id is not None:
			try:
				update_id = int(update_id)
				newest_update_id = update_id if newest_update_id is None else max(int(newest_update_id), update_id)
			except Exception:
				update_id = None

		cq = u.get("callback_query")
		if isinstance(cq, dict) and cq:
			commands.append(
				{
					"update_id": update_id,
					"text": f"__callback__:{str(cq.get('data', '') or '')}:{str(cq.get('id', '') or '')}:{cq.get('message', {}).get('message_id')}",
					"chat_id": cq.get("message", {}).get("chat", {}).get("id"),
					"user_id": cq.get("from", {}).get("id"),
					"username": str(cq.get("from", {}).get("username", "") or ""),
					"first_name": str(cq.get("from", {}).get("first_name", "") or ""),
					"chat_type": str(cq.get("message", {}).get("chat", {}).get("type", "") or ""),
				}
			)
			continue

		message = u.get("message", {})
		chat_type = str(message.get("chat", {}).get("type", "") or "")
		from_user = message.get("from", {}) if isinstance(message.get("from"), dict) else {}
		user_id = from_user.get("id")
		username = str(from_user.get("username", "") or "")
		first_name = str(from_user.get("first_name", "") or "")
		web_app_data = message.get("web_app_data") if isinstance(message, dict) else None
		if isinstance(web_app_data, dict):
			raw_web_app_data = str(web_app_data.get("data", "") or "").strip()
			action = ""
			if raw_web_app_data:
				try:
					parsed = json.loads(raw_web_app_data)
					if isinstance(parsed, dict):
						action = str(parsed.get("action", "") or "").strip()
				except Exception:
					action = raw_web_app_data
			if action:
				commands.append(
					{
						"update_id": update_id,
						"text": f"{webapp_command_prefix}{action}",
						"chat_id": message.get("chat", {}).get("id"),
						"user_id": user_id,
						"username": username,
						"first_name": first_name,
						"chat_type": chat_type,
					}
				)
				continue

		commands.append(
			{
				"update_id": update_id,
				"text": str(message.get("text", "") or ""),
				"chat_id": message.get("chat", {}).get("id"),
				"user_id": user_id,
				"username": username,
				"first_name": first_name,
				"chat_type": chat_type,
			}
		)

	return commands, newest_update_id
