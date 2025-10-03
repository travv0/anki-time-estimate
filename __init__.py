"""Anki add-on: FSRS-based time estimates on deck and overview screens."""

from __future__ import annotations

import html
import os
from typing import Any

from aqt import gui_hooks
from aqt.deckbrowser import DeckBrowser
from aqt.webview import WebContent
from aqt.overview import Overview

from .estimator import FsrsTimeEstimator, format_seconds


def _addon_config(mw) -> dict[str, Any]:
    try:
        return mw.addonManager.getConfig(__name__) or {}
    except Exception:
        return {}


def _is_debug_enabled(mw) -> bool:
    if os.environ.get("FSRS_TIME_DEBUG"):
        return True
    config = _addon_config(mw)
    return bool(config.get("debug"))


def _format_debug_block(text: str, title: str) -> str:
    safe_title = html.escape(title)
    escaped = html.escape(text)
    return (
        "<details class='fsrs-time-debug'>"
        f"<summary>{safe_title}</summary>"
        f"<pre>{escaped}</pre>"
        "</details>"
    )


def _deck_browser_stats(deck_browser: DeckBrowser, content) -> None:
    col = getattr(deck_browser.mw, "col", None)
    if not col:
        return
    debug_enabled = _is_debug_enabled(deck_browser.mw)
    try:
        estimator = FsrsTimeEstimator(col, debug=debug_enabled)
        result = estimator.estimate_for_all_decks()
    except Exception as err:  # pragma: no cover
        import traceback

        traceback.print_exc()
        print(f'[FSRS time estimate] deck browser error: {err!r}')
        return
    formatted_total = format_seconds(max(result.total_seconds, 0.0))
    formatted_new = format_seconds(max(result.new_seconds, 0.0))
    formatted_learn = format_seconds(max(result.learn_seconds, 0.0))
    formatted_review = format_seconds(max(result.review_seconds, 0.0))
    label = (
        "<div class='fsrs-time-estimate'>"
        f"FSRS workload across decks: {formatted_total} "
        f"(new: {formatted_new}, learn: {formatted_learn}, review: {formatted_review})"
        "</div>"
    )
    content.stats = label + (content.stats or "")

    if debug_enabled:
        debug_report = estimator.consume_logs()
        if debug_report:
            content.stats += _format_debug_block(debug_report, "FSRS debug – deck browser")

def _overview_stats(overview: Overview, content) -> None:
    col = getattr(overview.mw, "col", None)
    if not col:
        return
    deck_id = col.decks.get_current_id()
    debug_enabled = _is_debug_enabled(overview.mw)
    try:
        estimator = FsrsTimeEstimator(col, debug=debug_enabled)
        result = estimator.estimate_for_deck(deck_id)
        if result is not None:
            formatted_total = format_seconds(max(result.total_seconds, 0.0))
            formatted_new = format_seconds(max(result.new_seconds, 0.0))
            formatted_learn = format_seconds(max(result.learn_seconds, 0.0))
            formatted_review = format_seconds(max(result.review_seconds, 0.0))
            snippet = (
                "<div class='fsrs-time-estimate'>"
                f"FSRS workload for this deck: {formatted_total} "
                f"(new: {formatted_new}, learn: {formatted_learn}, review: {formatted_review})"
                "</div>"
            )
            content.table = (content.table or "") + snippet
    except Exception as err:  # pragma: no cover
        import traceback

        traceback.print_exc()
        print(f'[FSRS time estimate] overview error: {err!r}')
        return

    if debug_enabled:
        debug_report = estimator.consume_logs()
        if debug_report:
            content.table = (content.table or "") + _format_debug_block(
                debug_report,
                "FSRS debug – deck overview",
            )

def _inject_css(web_content: WebContent, context: object | None = None) -> None:
    style = (
        ".fsrs-time-estimate {"
        "margin-top: 0.5em;"
        "font-size: 0.9em;"
        "color: var(--fg subtle, #555);"
        "}"
        ".fsrs-time-debug {"
        "margin-top: 0.5em;"
        "font-size: 0.8em;"
        "color: var(--fg subtle, #666);"
        "}"
        ".fsrs-time-debug pre {"
        "white-space: pre-wrap;"
        "margin: 0.5em 0 0;"
        "}"
    )
    web_content.head += f"<style>{style}</style>"


def _init() -> None:
    gui_hooks.deck_browser_will_render_content.append(_deck_browser_stats)
    gui_hooks.overview_will_render_content.append(_overview_stats)
    gui_hooks.webview_will_set_content.append(_inject_css)


_init()
