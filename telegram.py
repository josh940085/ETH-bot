import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

try:
	from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - urllib3 variant fallback
	NotOpenSSLWarning = None

if NotOpenSSLWarning is not None:
	warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from cryptography.fernet import Fernet, InvalidToken

try:
	import fcntl
except ImportError:  # pragma: no cover - Windows fallback
	fcntl = None

from n8n_client import post_n8n_notification
from runtime_config import is_truthy as _is_truthy
from runtime_paths import data_path


def _safe_float(value, default=0.0):
	try:
		return float(value)
	except Exception:
		return default


TELEGRAM_STATE_FILE = data_path(".telegram_state.json")
TELEGRAM_SECRET_DIR = Path(
	str(os.getenv("BOT_SECRET_DIR", TELEGRAM_STATE_FILE.parent.parent / "secrets") or "")
).expanduser()
TELEGRAM_STATE_KEY_FILE = TELEGRAM_SECRET_DIR / "telegram_state.key"
TELEGRAM_SEND_LOCK_FILE = TELEGRAM_STATE_FILE.parent / ".telegram_send_rate"
TELEGRAM_STATE_ENCRYPTION_PREFIX = "tgstate:v1:"
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

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
DISCORD_NEWS = os.getenv("DISCORD_NEWS", "")
DISCORD_AUTO_DELETE_HOURS = max(0.0, _safe_float(os.getenv("DISCORD_AUTO_DELETE_HOURS", 24.0), 24.0))
DISCORD_AUTO_DELETE_SEC = int(DISCORD_AUTO_DELETE_HOURS * 3600)

LAST_TELEGRAM_TS = 0

_control_panel_builder = None


def _chmod_private(path):
	try:
		Path(path).chmod(0o600)
	except Exception:
		pass


def _load_or_create_state_key() -> bytes:
	TELEGRAM_STATE_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
	try:
		TELEGRAM_STATE_KEY_FILE.parent.chmod(0o700)
	except Exception:
		pass
	try:
		key = TELEGRAM_STATE_KEY_FILE.read_bytes().strip()
	except FileNotFoundError:
		key = Fernet.generate_key()
		try:
			fd = os.open(
				str(TELEGRAM_STATE_KEY_FILE),
				os.O_WRONLY | os.O_CREAT | os.O_EXCL,
				0o600,
			)
		except FileExistsError:
			key = TELEGRAM_STATE_KEY_FILE.read_bytes().strip()
		else:
			with os.fdopen(fd, "wb") as fh:
				fh.write(key)
				fh.flush()
				os.fsync(fh.fileno())
	_chmod_private(TELEGRAM_STATE_KEY_FILE)
	# Fernet validates the key length and encoding.
	Fernet(key)
	return key


def _encode_telegram_state(payload: dict) -> str:
	raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
	token = Fernet(_load_or_create_state_key()).encrypt(raw).decode("ascii")
	return TELEGRAM_STATE_ENCRYPTION_PREFIX + token


def _decode_telegram_state(raw: str) -> dict:
	text = str(raw or "").strip()
	if not text:
		return {}
	if text.startswith(TELEGRAM_STATE_ENCRYPTION_PREFIX):
		token = text[len(TELEGRAM_STATE_ENCRYPTION_PREFIX):].encode("ascii")
		try:
			text = Fernet(_load_or_create_state_key()).decrypt(token).decode("utf-8")
		except (InvalidToken, UnicodeDecodeError) as exc:
			raise ValueError("encrypted Telegram state could not be decrypted") from exc
	payload = json.loads(text)
	if not isinstance(payload, dict):
		raise ValueError("Telegram state must be a JSON object")
	return payload


def configure_token(token=None):
	global TELEGRAM_TOKEN
	if token is None:
		TELEGRAM_TOKEN = str(os.getenv("TELEGRAM_TOKEN", "") or "").strip()
	else:
		TELEGRAM_TOKEN = str(token or "").strip()
	ensure_telegram_state_encrypted()
	return TELEGRAM_TOKEN


def parse_telegram_state(raw: str) -> dict:
	try:
		return _decode_telegram_state(raw)
	except Exception:
		return {}


def ensure_telegram_state_encrypted() -> bool:
	if not TELEGRAM_STATE_FILE.exists():
		return True
	try:
		raw = TELEGRAM_STATE_FILE.read_text(encoding="utf-8")
		if raw.strip().startswith(TELEGRAM_STATE_ENCRYPTION_PREFIX):
			_decode_telegram_state(raw)
			_chmod_private(TELEGRAM_STATE_FILE)
			return True
		payload = _decode_telegram_state(raw)
		temp_path = TELEGRAM_STATE_FILE.with_name(TELEGRAM_STATE_FILE.name + ".encrypting")
		temp_path.write_text(_encode_telegram_state(payload), encoding="utf-8")
		_chmod_private(temp_path)
		os.replace(temp_path, TELEGRAM_STATE_FILE)
		return True
	except Exception as exc:
		print(f"⚠️ Telegram state 加密遷移失敗: {exc}")
		return False


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
		return _decode_telegram_state(raw)
	except Exception:
		return {}


def update_telegram_state(mutator):
	try:
		TELEGRAM_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
		if not TELEGRAM_STATE_FILE.exists():
			try:
				fd = os.open(
					str(TELEGRAM_STATE_FILE),
					os.O_WRONLY | os.O_CREAT | os.O_EXCL,
					0o600,
				)
			except FileExistsError:
				pass
			else:
				os.close(fd)
		with TELEGRAM_STATE_FILE.open("a+", encoding="utf-8") as fh:
			if fcntl is not None:
				fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

			fh.seek(0)
			raw = fh.read()
			try:
				payload = _decode_telegram_state(raw)
			except Exception as exc:
				print(f"⚠️ Telegram state 解密失敗，拒絕覆寫: {exc}")
				return None
			result = mutator(payload)

			fh.seek(0)
			fh.truncate()
			fh.write(_encode_telegram_state(payload))
			fh.flush()
			os.fsync(fh.fileno())
			_chmod_private(TELEGRAM_STATE_FILE)

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


def get_allowed_telegram_user_ids() -> set:
	allowed = set()
	for env_name in (
		"POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS",
		"TELEGRAM_PRIVATE_CHAT_ID",
		"TELEGRAM_CHAT_ID",
	):
		raw = str(os.getenv(env_name, "") or "")
		for item in re.split(r"[\s,;]+", raw):
			try:
				value = int(item.strip())
			except Exception:
				continue
			if value > 0:
				allowed.add(value)
	return allowed


def is_authorized_telegram_user(chat_id=None, user_id=None, chat_type="") -> bool:
	if str(chat_type or "").strip().lower() != "private":
		return False
	if not is_private_chat_id(chat_id):
		return False
	try:
		resolved_user_id = int(str(user_id if user_id not in (None, "") else chat_id).strip())
		resolved_chat_id = int(str(chat_id).strip())
	except Exception:
		return False
	allowed = get_allowed_telegram_user_ids()
	return bool(allowed and resolved_user_id == resolved_chat_id and resolved_user_id in allowed)


def _notification_opt_out_ids(payload=None) -> set:
	state = payload if isinstance(payload, dict) else read_telegram_state_locked()
	result = set()
	for item in state.get("notification_opt_out_ids", []) if isinstance(state, dict) else []:
		try:
			value = int(str(item).strip())
		except Exception:
			continue
		if value > 0:
			result.add(value)
	return result


def set_notification_opt_out(chat_id, opted_out=True):
	try:
		chat_num = int(str(chat_id).strip())
	except Exception:
		return False
	if chat_num <= 0 or chat_num not in get_allowed_telegram_user_ids():
		return False

	def _mutate(payload):
		ids = _notification_opt_out_ids(payload)
		if opted_out:
			ids.add(chat_num)
		else:
			ids.discard(chat_num)
		payload["notification_opt_out_ids"] = sorted(ids)
		return True

	return bool(update_telegram_state(_mutate))


def delete_telegram_user_data(chat_id, user_id=None) -> bool:
	try:
		chat_text = str(int(str(chat_id).strip()))
		user_text = str(int(str(user_id if user_id not in (None, "") else chat_id).strip()))
	except Exception:
		return False

	def _belongs(item):
		if not isinstance(item, dict):
			return False
		return normalize_chat_id(item.get("chat_id")) == chat_text or normalize_chat_id(item.get("user_id")) == user_text

	def _mutate(payload):
		for key in ("notify_chat_ids", "notification_opt_out_ids"):
			items = payload.get(key)
			if isinstance(items, list):
				payload[key] = [
					item for item in items
					if normalize_chat_id(item) not in {chat_text, user_text}
				]
		if normalize_chat_id(payload.get("last_private_chat_id")) == chat_text:
			payload.pop("last_private_chat_id", None)
		for key in ("pending_commands", "telegram_delivery_events"):
			items = payload.get(key)
			if isinstance(items, list):
				payload[key] = [item for item in items if not _belongs(item)]
		summary = payload.get("telegram_delivery_summary")
		if isinstance(summary, dict) and normalize_chat_id(summary.get("last_error_chat_id")) == chat_text:
			payload.pop("telegram_delivery_summary", None)
		return True

	return bool(update_telegram_state(_mutate))


def _remember_notification_chat_mutate(payload, chat_id):
	chat_text = normalize_chat_id(chat_id)
	if not chat_text:
		return

	try:
		chat_num = int(chat_text)
	except Exception:
		return

	if chat_num <= 0 or chat_num not in get_allowed_telegram_user_ids():
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

	payload = read_telegram_state_locked()
	allowed = get_allowed_telegram_user_ids()
	opted_out = _notification_opt_out_ids(payload)

	def _append(value):
		chat_text = normalize_chat_id(value)
		if not chat_text or chat_text in seen:
			return
		try:
			chat_num = int(chat_text)
		except Exception:
			return
		if chat_num not in allowed or chat_num in opted_out:
			return
		seen.add(chat_text)
		targets.append(chat_text)

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


def truncate_text(value, limit=220):
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
		description = truncate_text(body)

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
		description = truncate_text(error)

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
		"description": truncate_text(info.get("description", "")),
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
			if int(candidate) in get_allowed_telegram_user_ids():
				return candidate
		except Exception:
			pass

	payload = read_telegram_state_locked()
	candidate = normalize_chat_id(payload.get("last_private_chat_id"))
	if candidate:
		try:
			if int(candidate) in get_allowed_telegram_user_ids():
				return candidate
		except Exception:
			pass

	for candidate_num in sorted(get_allowed_telegram_user_ids()):
		return str(candidate_num)
	return None


def _append_pending_command(chat_id, text, update_id, user_id=None, username="", first_name="", chat_type=""):
	def _mutate(payload):
		queue = payload.get("pending_commands")
		if not isinstance(queue, list):
			queue = []

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
	"""Single-shot send with delivery tracking, for callers that don't need
	sanitization/control-panel markup/retry (see _send_telegram_message for
	that). Shares _post_telegram_message as the one place that actually
	talks to Telegram (n8n-routed with a direct-API fallback), rather than
	making its own separate HTTP call.
	"""
	if not str(token or TELEGRAM_TOKEN or "").strip() or chat_id is None:
		return None

	try:
		response = _post_telegram_message(chat_id, text, timeout=timeout, token=token)
		info = note_telegram_delivery_event(
			chat_id=chat_id,
			ok=response is not None and response.status_code == 200,
			status_code=getattr(response, "status_code", "no-response"),
			body=getattr(response, "text", ""),
			error="sendMessage returned no response" if response is None else None,
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


def wait_for_telegram_send_slot(min_interval_sec=1.05):
	"""Enforce Telegram's per-chat guidance conservatively across local processes."""
	interval = max(1.0, float(min_interval_sec or 1.05))
	TELEGRAM_SEND_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
	with TELEGRAM_SEND_LOCK_FILE.open("a+", encoding="utf-8") as fh:
		_chmod_private(TELEGRAM_SEND_LOCK_FILE)
		if fcntl is not None:
			fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
		fh.seek(0)
		try:
			last_ts = float((fh.read() or "0").strip())
		except Exception:
			last_ts = 0.0
		wait_sec = interval - (time.time() - last_ts)
		if wait_sec > 0:
			time.sleep(wait_sec)
		fh.seek(0)
		fh.truncate()
		fh.write(f"{time.time():.6f}")
		fh.flush()
		os.fsync(fh.fileno())
		if fcntl is not None:
			fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _mark_update_consumed(update_id):
	def _mutate(payload):
		payload["last_update_id"] = int(update_id)

	update_telegram_state(_mutate)


def _is_telegram_poll_conflict_error(err) -> bool:
	text = str(err or "")
	return "409" in text and "Conflict" in text


def _is_telegram_poll_timeout_error(err) -> bool:
	if isinstance(err, requests.exceptions.Timeout):
		return True
	text = str(err or "")
	return "Read timed out" in text or "read timeout" in text.lower()


def _telegram_poll_retry_after(err) -> int:
	response = getattr(err, "response", None)
	if response is None:
		return 0
	try:
		payload = response.json()
		parameters = payload.get("parameters", {}) if isinstance(payload, dict) else {}
		return max(0, int(parameters.get("retry_after", 0) or 0))
	except Exception:
		try:
			return max(0, int(response.headers.get("Retry-After", 0) or 0))
		except Exception:
			return 0


def _redact_telegram_error(err) -> str:
	text = str(err or "")
	return re.sub(r"/bot[^/\s]+/", "/bot<redacted>/", text)


def _note_telegram_poll_success():
	global TELEGRAM_POLL_BACKOFF_SEC, TELEGRAM_POLL_LAST_ERROR_KEY, TELEGRAM_POLL_LAST_LOG_TS
	TELEGRAM_POLL_BACKOFF_SEC = 1.0
	TELEGRAM_POLL_LAST_ERROR_KEY = ""
	TELEGRAM_POLL_LAST_LOG_TS = 0.0


def _handle_telegram_poll_error(err):
	global TELEGRAM_POLL_BACKOFF_SEC, TELEGRAM_POLL_LAST_ERROR_KEY, TELEGRAM_POLL_LAST_LOG_TS

	now_ts = time.time()
	text = _redact_telegram_error(err)
	is_conflict = _is_telegram_poll_conflict_error(err)
	is_timeout = _is_telegram_poll_timeout_error(err)
	retry_after = _telegram_poll_retry_after(err)
	is_rate_limit = retry_after > 0 or "429" in text or "too many requests" in text.lower()

	if is_conflict:
		backoff = max(15.0, TELEGRAM_POLL_BACKOFF_SEC)
		next_backoff = max(backoff * 2, 30.0)
		TELEGRAM_POLL_BACKOFF_SEC = min(TELEGRAM_POLL_BACKOFF_MAX, next_backoff)
	elif is_timeout:
		backoff = min(TELEGRAM_POLL_TIMEOUT_BACKOFF_MAX, max(1.0, TELEGRAM_POLL_BACKOFF_SEC))
		next_backoff = max(backoff * 1.6, 2.0)
		TELEGRAM_POLL_BACKOFF_SEC = min(TELEGRAM_POLL_TIMEOUT_BACKOFF_MAX, next_backoff)
	elif is_rate_limit:
		backoff = max(float(retry_after), TELEGRAM_POLL_BACKOFF_SEC, 2.0)
		TELEGRAM_POLL_BACKOFF_SEC = min(TELEGRAM_POLL_BACKOFF_MAX, max(backoff * 2, 5.0))
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
	elif is_rate_limit:
		error_key = "telegram-429-rate-limit"
		message = f"⚠️ Telegram 輪詢觸發 429，依 retry_after 等待 {backoff:.0f}s 後重試"
		min_log_interval = max(backoff, 30.0)
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
			cq_from = cq.get("from", {}) if isinstance(cq.get("from"), dict) else {}
			cq_chat_type = str(cq_msg.get("chat", {}).get("type", "") or "")
			if not is_authorized_telegram_user(
				cq_chat_id,
				cq_from.get("id"),
				cq_chat_type,
			):
				_mark_update_consumed(update_id)
				continue

			_append_pending_callback(
				cq_chat_id,
				cq_data,
				cq_id,
				cq_msg_id,
				update_id,
				user_id=cq_from.get("id"),
				username=str(cq_from.get("username", "") or ""),
				first_name=str(cq_from.get("first_name", "") or ""),
				chat_type=cq_chat_type,
			)
			continue

		if not is_authorized_telegram_user(chat_id, user_id, chat_type):
			_mark_update_consumed(update_id)
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
			if not is_authorized_telegram_user(
				item.get("chat_id"),
				item.get("user_id"),
				item.get("chat_type"),
			):
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
			cq_message = cq.get("message", {}) if isinstance(cq.get("message"), dict) else {}
			cq_chat = cq_message.get("chat", {}) if isinstance(cq_message.get("chat"), dict) else {}
			cq_from = cq.get("from", {}) if isinstance(cq.get("from"), dict) else {}
			if not is_authorized_telegram_user(
				cq_chat.get("id"),
				cq_from.get("id"),
				cq_chat.get("type"),
			):
				continue
			commands.append(
				{
					"update_id": update_id,
					"text": f"__callback__:{str(cq.get('data', '') or '')}:{str(cq.get('id', '') or '')}:{cq_message.get('message_id')}",
					"chat_id": cq_chat.get("id"),
					"user_id": cq_from.get("id"),
					"username": str(cq_from.get("username", "") or ""),
					"first_name": str(cq_from.get("first_name", "") or ""),
					"chat_type": str(cq_chat.get("type", "") or ""),
				}
			)
			continue

		message = u.get("message", {})
		chat_type = str(message.get("chat", {}).get("type", "") or "")
		from_user = message.get("from", {}) if isinstance(message.get("from"), dict) else {}
		user_id = from_user.get("id")
		username = str(from_user.get("username", "") or "")
		first_name = str(from_user.get("first_name", "") or "")
		chat_id = message.get("chat", {}).get("id")
		if not is_authorized_telegram_user(chat_id, user_id, chat_type):
			continue
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
						"chat_id": chat_id,
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
				"chat_id": chat_id,
				"user_id": user_id,
				"username": username,
				"first_name": first_name,
				"chat_type": chat_type,
			}
		)

	return commands, newest_update_id


def set_control_panel_builder(fn):
	"""Register the caller's trading-bot-specific control-panel keyboard builder.

	telegram.py has no notion of trading state (active_trade,
	POSITION_PANEL_STATE); the caller (eth.py) registers a callback here
	instead of telegram.py importing back from eth.py, which would be
	circular.
	"""
	global _control_panel_builder
	_control_panel_builder = fn


def _sanitize_telegram_text(msg):
	raw = str(msg or "")
	# Telegram sendMessage is sent without parse_mode here. Preserve ampersands
	# because signed panel URLs rely on query separators such as &state_url=.
	safe_text = raw.replace("<", "").replace(">", "")
	fallback_text = raw.replace("<", "").replace(">", "")
	return safe_text, fallback_text


def _post_telegram_message(chat_id, text, reply_markup=None, timeout=5, token=None):
	"""The one place that actually calls Telegram's sendMessage: n8n-routed
	first, direct API as fallback. Every higher-level sender (send_message,
	_send_telegram_message and everything built on it) goes through this
	instead of making its own HTTP call.
	"""
	resolved_token = str(token or TELEGRAM_TOKEN or "").strip()
	if not resolved_token or chat_id is None:
		return None

	payload = {
		"chat_id": chat_id,
		"text": text,
	}
	if reply_markup is not None:
		payload["reply_markup"] = reply_markup

	wait_for_telegram_send_slot()
	# n8n routing relies on n8n's own server-side configured token (see
	# n8n_service.py's ALLOWED_NOTIFICATION_SECRETS) rather than `token` -
	# an explicit token override here only fully applies to the direct
	# fallback below. In practice every caller's override already matches
	# the module-level TELEGRAM_TOKEN, so this doesn't diverge today.
	n8n_response = post_n8n_notification(
		"telegram",
		payload,
		timeout=timeout,
		session=HTTP_SESSION,
	)
	if n8n_response is not None:
		return n8n_response

	try:
		return HTTP_SESSION.post(
			f"https://api.telegram.org/bot{resolved_token}/sendMessage",
			json=payload,
			timeout=timeout,
		)
	except Exception:
		return None


def _send_telegram_message(chat_id, msg, include_control_panel=False, timeout=5):
	safe_text, fallback_text = _sanitize_telegram_text(msg)
	reply_markup = (
		_control_panel_builder(chat_id)
		if include_control_panel and _control_panel_builder is not None and is_private_chat_id(chat_id)
		else None
	)

	res = _post_telegram_message(chat_id, safe_text, reply_markup=reply_markup, timeout=timeout)
	if res is not None and res.status_code == 400 and fallback_text != safe_text:
		res = _post_telegram_message(chat_id, fallback_text, reply_markup=reply_markup, timeout=timeout)
	return res


def _is_telegram_chat_not_found(status, body) -> bool:
	try:
		status_int = int(status)
	except (TypeError, ValueError):
		status_int = 0
	if status_int != 400:
		return False

	text = str(body or "")
	low = text.lower()
	return "chat not found" in low or '"description":"bad request: chat not found"' in low


def send_telegram(msg, priority=False, include_private=True):
	global LAST_TELEGRAM_TS

	now = time.time()

	# ===== 只有低優先才限流 =====
	if not priority and now - LAST_TELEGRAM_TS < 10:
		return False

	if include_private:
		targets = get_notification_chat_ids()
	else:
		targets = []  # 群聊推播已停用

	if not targets and _is_truthy(os.getenv("TELEGRAM_FALLBACK_PRIVATE_WHEN_NO_BROADCAST_TARGET", "1")):
		private_target = resolve_private_chat_id_for_controls()
		if private_target:
			targets = [private_target]

	if not TELEGRAM_TOKEN or not targets:
		print("⚠️ Telegram 目標未設定，略過發送")
		return False

	try:
		sent_count = 0
		for chat_id in targets:
			res = _send_telegram_message(chat_id, msg, include_control_panel=True)

			if res is None or res.status_code != 200:
				status = getattr(res, "status_code", "no-response")
				body = getattr(res, "text", "")
				delivery = note_telegram_delivery_event(
					chat_id=chat_id,
					ok=False,
					status_code=status,
					body=body,
					error="sendMessage returned no response" if res is None else None,
					context="telegram.send_telegram.broadcast",
				)
				print(f"❌ Telegram 發送失敗 [{chat_id}]:", status, body)

				if delivery.get("remove_chat") or _is_telegram_chat_not_found(status, body):
					remove_notification_chat(chat_id)
					continue

				try:
					retry_after = max(1, int(delivery.get("retry_after", 0) or 0))
					time.sleep(min(30, retry_after))
					res2 = _send_telegram_message(chat_id, msg, include_control_panel=True)
					retry_status = getattr(res2, "status_code", "no-response")
					retry_body = getattr(res2, "text", "")
					retry_delivery = note_telegram_delivery_event(
						chat_id=chat_id,
						ok=res2 is not None and res2.status_code == 200,
						status_code=retry_status,
						body=retry_body,
						error="retry sendMessage returned no response" if res2 is None else None,
						context="telegram.send_telegram.broadcast_retry",
					)
					print(f"🔁 retry [{chat_id}]:", retry_status)
					if res2 is not None and res2.status_code == 200:
						sent_count += 1
					elif retry_delivery.get("remove_chat"):
						remove_notification_chat(chat_id)
				except Exception as e:
					note_telegram_delivery_event(
						chat_id=chat_id,
						ok=False,
						status_code="exception",
						error=e,
						context="telegram.send_telegram.broadcast_retry",
					)
					print(f"❌ retry失敗 [{chat_id}]:", e)
			else:
				note_telegram_delivery_event(
					chat_id=chat_id,
					ok=True,
					status_code=res.status_code,
					body=getattr(res, "text", ""),
					context="telegram.send_telegram.broadcast",
				)
				sent_count += 1

		if sent_count > 0:
			print(f"✅ Telegram 已送出 ({sent_count}/{len(targets)})")

		# Discord只發「進場通知」
		try:
			if DISCORD_WEBHOOK and "進場" in msg:
				_post_discord_webhook(DISCORD_WEBHOOK, msg, timeout=5)
		except Exception as e:
			print("Discord error:", e)

		LAST_TELEGRAM_TS = now

	except Exception as e:
		print("❌ Telegram error:", e, "| msg:", msg[:50])
		return False

	return sent_count > 0


def send_private_telegram(msg, priority=False):
	global LAST_TELEGRAM_TS

	now = time.time()
	if not priority and now - LAST_TELEGRAM_TS < 10:
		return False

	dedupe_key = ""
	dedupe_cache = {}
	if _is_truthy(os.getenv("TELEGRAM_PRIVATE_DEDUPE_ENABLED", "1")):
		dedupe_text = str(msg or "").strip()
		if dedupe_text:
			dedupe_key = hashlib.sha256(dedupe_text.encode("utf-8", errors="ignore")).hexdigest()
			dedupe_cache = getattr(send_private_telegram, "_dedupe_cache", {})
			cooldown = max(
				30.0,
				_safe_float(
					os.getenv(
						"TELEGRAM_PRIVATE_PRIORITY_DEDUPE_SEC" if priority else "TELEGRAM_PRIVATE_DEDUPE_SEC",
						180 if priority else 60,
					),
					180 if priority else 60,
				),
			)
			last_sent = _safe_float(dedupe_cache.get(dedupe_key), 0.0)
			if now - last_sent < cooldown:
				if now - _safe_float(getattr(send_private_telegram, "_last_dedupe_log_ts", 0.0), 0.0) > 60:
					print("🔕 私聊重複通知已略過")
					send_private_telegram._last_dedupe_log_ts = now
				return False

	target = resolve_private_chat_id_for_controls()
	if not TELEGRAM_TOKEN or not target:
		print("⚠️ 私聊目標未設定，略過發送")
		return False

	try:
		res = _send_telegram_message(target, msg, include_control_panel=True)

		if res is None or res.status_code != 200:
			status = getattr(res, "status_code", "no-response")
			body = getattr(res, "text", "")
			delivery = note_telegram_delivery_event(
				chat_id=target,
				ok=False,
				status_code=status,
				body=body,
				error="sendMessage returned no response" if res is None else None,
				context="telegram.send_private_telegram",
			)
			print(f"❌ 私聊發送失敗 [{target}]", status, body)

			if delivery.get("remove_chat") or _is_telegram_chat_not_found(status, body):
				remove_notification_chat(target)

			return False

		note_telegram_delivery_event(
			chat_id=target,
			ok=True,
			status_code=res.status_code,
			body=getattr(res, "text", ""),
			context="telegram.send_private_telegram",
		)
		LAST_TELEGRAM_TS = now
		if dedupe_key:
			dedupe_cache[dedupe_key] = now
			if len(dedupe_cache) > 300:
				cutoff = now - 3600.0
				dedupe_cache = {key: ts for key, ts in dedupe_cache.items() if _safe_float(ts, 0.0) >= cutoff}
			send_private_telegram._dedupe_cache = dedupe_cache
		summary = str(msg or "").strip().splitlines()[0][:48]
		print(f"✅ 私聊通知已送出: {summary}" if summary else "✅ 私聊通知已送出")
		return True
	except Exception as e:
		note_telegram_delivery_event(
			chat_id=target,
			ok=False,
			status_code="exception",
			error=e,
			context="telegram.send_private_telegram",
		)
		print("❌ 私聊通知錯誤:", e)
		return False


def _send_trade_notification(msg, priority=True):
	delivered = send_telegram(msg, priority=priority)
	if delivered:
		return True
	return send_private_telegram(msg, priority=priority)


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
