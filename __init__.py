"""Anki add-on: FSRS-based time estimates on deck and overview screens."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Any

from aqt import gui_hooks
from aqt.deckbrowser import DeckBrowser
from aqt.overview import Overview
from aqt.webview import WebContent

from .estimator import EstimateResult, FsrsTimeEstimator, format_seconds


@dataclass
class _DeckBrowserCache:
    signature: tuple[tuple[int, int, int, int, int, int, int], ...]
    totals: EstimateResult
    per_deck: dict[int, EstimateResult]


_CACHE_ATTR = "_fsrs_time_cache"


def _report_error(label: str, err: Exception) -> None:
    import traceback

    traceback.print_exc()
    print(f"[FSRS time estimate] {label} error: {err!r}")


def _deck_tree_signature(col, tree) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
    decks = col.decks
    signature: list[tuple[int, int, int, int, int, int, int]] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        deck_id = int(getattr(node, "deck_id", 0))
        deck_dict = decks.get(deck_id, default=False) or {}
        conf_id = int(deck_dict.get("conf", 0))
        mod = int(deck_dict.get("mod", 0))
        new_count = int(getattr(node, "new_count", 0))
        learn_count = int(getattr(node, "learn_count", 0))
        review_count = int(getattr(node, "review_count", 0))
        interday_raw = float(
            getattr(
                node,
                "interday_learning",
                getattr(node, "interday_learning_uncapped", 0.0),
            )
        )
        interday_count = int(round(interday_raw))
        signature.append(
            (
                deck_id,
                new_count,
                learn_count,
                review_count,
                interday_count,
                conf_id,
                mod,
            )
        )
        for child in getattr(node, "children", []) or []:
            stack.append(child)
    signature.sort()
    return tuple(signature)


def _sum_estimates(a: EstimateResult, b: EstimateResult) -> EstimateResult:
    return EstimateResult(
        total_seconds=a.total_seconds + b.total_seconds,
        new_seconds=a.new_seconds + b.new_seconds,
        learn_seconds=a.learn_seconds + b.learn_seconds,
        review_seconds=a.review_seconds + b.review_seconds,
    )


def _compute_tree_estimates(estimator: FsrsTimeEstimator, tree) -> tuple[EstimateResult, dict[int, EstimateResult]]:
    totals = EstimateResult(0.0, 0.0, 0.0, 0.0)
    per_deck: dict[int, EstimateResult] = {}
    for child in getattr(tree, "children", []) or []:
        result = estimator.estimate_for_deck_node(child)
        per_deck[int(child.deck_id)] = result
        totals = _sum_estimates(totals, result)
    return totals, per_deck


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
        tree = col.sched.deck_due_tree()
    except Exception as err:  # pragma: no cover
        _report_error("deck browser", err)
        return

    estimator: FsrsTimeEstimator | None = None
    totals: EstimateResult
    per_deck: dict[int, EstimateResult]

    if not debug_enabled:
        signature = _deck_tree_signature(col, tree)
        cache_entry: _DeckBrowserCache | None = getattr(deck_browser.mw, _CACHE_ATTR, None)
        if cache_entry and cache_entry.signature == signature:
            totals = cache_entry.totals
            per_deck = cache_entry.per_deck
        else:
            try:
                estimator = FsrsTimeEstimator(col, debug=False)
                totals, per_deck = _compute_tree_estimates(estimator, tree)
            except Exception as err:  # pragma: no cover
                _report_error("deck browser", err)
                return
            setattr(
                deck_browser.mw,
                _CACHE_ATTR,
                _DeckBrowserCache(signature=signature, totals=totals, per_deck=per_deck),
            )
    else:
        try:
            estimator = FsrsTimeEstimator(col, debug=True)
            totals, per_deck = _compute_tree_estimates(estimator, tree)
        except Exception as err:  # pragma: no cover
            _report_error("deck browser", err)
            return

    result = totals
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

    if debug_enabled and estimator is not None:
        debug_report = estimator.consume_logs()
        if debug_report:
            content.stats += _format_debug_block(debug_report, "FSRS debug – deck browser")

def _overview_stats(overview: Overview, content) -> None:
    col = getattr(overview.mw, "col", None)
    if not col:
        return
    deck_id = col.decks.get_current_id()
    debug_enabled = _is_debug_enabled(overview.mw)
    cache_entry: _DeckBrowserCache | None = getattr(overview.mw, _CACHE_ATTR, None)
    result: EstimateResult | None = None

    if cache_entry and not debug_enabled:
        result = cache_entry.per_deck.get(int(deck_id))

    estimator: FsrsTimeEstimator | None = None
    if result is None:
        try:
            estimator = FsrsTimeEstimator(col, debug=debug_enabled)
            result = estimator.estimate_for_deck(deck_id)
        except Exception as err:  # pragma: no cover
            _report_error("overview", err)
            return

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

    if debug_enabled and estimator is not None:
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
