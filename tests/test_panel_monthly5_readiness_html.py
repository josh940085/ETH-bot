from pathlib import Path


PANEL_HTML = Path("docs/index.html")


def test_monthly5_panel_uses_promotion_ready_for_takeover_label():
    html = PANEL_HTML.read_text(encoding="utf-8")

    assert "const promotionReady = Boolean(state.promotion_ready);" in html
    assert 'const minSpanHours = Number.isFinite(Number(state.min_span_hours)) ? Number(state.min_span_hours) : 8;' in html
    assert "const promotionEtaTs = Number.isFinite(Number(state.promotion_earliest_review_ts))" in html
    assert "const promotionEtaLabel = promotionEtaTs > 0 ? ts_full_str(promotionEtaTs) : \"-\";" in html
    assert 'const tone = promotionReady ? "good"' in html
    assert 'const label = promotionReady ? "可接管"' in html
    assert 'status === "ready" || ready ? "待升級" : "收集中"' in html
    assert 'eta ${escapeHtml(promotionEtaLabel)} span-${spanRemainingHours.toFixed(2)}h' in html
