"""FSRS-based time estimation helpers for Anki deck workloads."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Iterable, List, Sequence

from anki.collection import Collection
from anki.decks import DeckId
from anki.scheduler_pb2 import SimulateFsrsReviewRequest


FsrsExposureCounts = namedtuple("FsrsExposureCounts", "new review")


def _ensure_list(values: Iterable[float] | None) -> list[float]:
    return [float(v) for v in values or []]


DEFAULT_FSRS_PARAMS = [
    0.4,
    0.6,
    2.4,
    6.5,
    1.3,
    4.5,
    0.22,
    1.7,
    2.0,
    5.2,
    0.6,
    1.0,
    0.18,
    0.3,
    0.4,
    0.5,
    0.6,
]

FALLBACK_NEW_SECONDS = 28.0
FALLBACK_REVIEW_SECONDS = 13.0


@dataclass
class AverageDurations:
    new_seconds: float
    review_seconds: float


@dataclass
class Contribution:
    name: str
    new_capacity: float
    learn_count: float
    interday_count: float
    review_capacity: float
    new_exposure_per_card: float
    review_exposure_per_card: float
    avg_new_seconds: float
    avg_review_seconds: float


@dataclass
class EstimateResult:
    total_seconds: float
    review_seconds: float


class FsrsTimeEstimator:
    """Estimate daily workload durations using FSRS configuration and history."""

    def __init__(self, col: Collection, debug: bool = False) -> None:
        self.col = col
        self.debug = debug
        self._logs: list[str] = []
        self._avg_cache: dict[int, tuple[int | None, AverageDurations]] = {}
        self._global_avg_cache: dict[tuple[int, str], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.debug:
            self._logs.append(message)

    def consume_logs(self) -> str:
        if not self.debug or not self._logs:
            return ""
        payload = "\n".join(self._logs)
        self._logs.clear()
        return payload

    @staticmethod
    def _deck_name(deck_node, deck_id: DeckId) -> str:
        name = getattr(deck_node, "name", "")
        if name:
            return name
        try:
            return str(int(deck_id))
        except Exception:
            return str(deck_id)

    def estimate_for_all_decks(self) -> EstimateResult:
        tree = self.col.sched.deck_due_tree()
        total = 0.0
        review_total = 0.0
        self._log("[all decks] starting aggregation")
        for child in tree.children:
            child_name = self._deck_name(child, child.deck_id)
            self._log(f"[all decks] processing {child_name} ({int(child.deck_id)})")
            result = self.estimate_for_deck(child.deck_id)
            if result:
                self._log(
                    f"[all decks] deck {child_name} => {result.total_seconds} seconds "
                    f"(reviews {result.review_seconds})"
                )
                total += result.total_seconds
                review_total += result.review_seconds
            else:
                self._log(f"[all decks] deck {child_name} => None")
        self._log(f"[all decks] total seconds {total} (reviews {review_total})")
        return EstimateResult(total_seconds=total, review_seconds=review_total)

    def estimate_for_deck(self, deck_id: DeckId) -> EstimateResult | None:
        deck_node = self.col.sched.deck_due_tree(deck_id)
        deck_name = self._deck_name(deck_node, deck_id)
        self._log(
            f"[deck] {deck_name} ({int(deck_id)}) counts new={deck_node.new_count} "
            f"learn={deck_node.learn_count} review={deck_node.review_count}"
        )

        total_due = deck_node.new_count + deck_node.learn_count + deck_node.review_count
        if total_due == 0:
            self._log(f"[deck] {deck_name} has no due cards; returning 0")
            return EstimateResult(0.0, 0.0)

        contributions = self._collect_contributions(deck_node)
        total, review = self._aggregate_contributions(deck_node, deck_name, contributions)
        self._log(
            f"[deck] {deck_name} total time {total:.2f}s (reviews {review:.2f}s) "
            f"from {len(contributions)} contributions"
        )
        return EstimateResult(total_seconds=total, review_seconds=review)

    def _collect_contributions(self, deck_node) -> List[Contribution]:
        deck_id = deck_node.deck_id
        deck = self.col.decks.get(deck_id, default=False) or {}
        deck_name = self._deck_name(deck_node, deck_id)

        child_nodes = list(getattr(deck_node, "children", []))
        contributions: List[Contribution] = []

        child_new = 0.0
        child_learn = 0.0
        child_review = 0.0
        child_interday = 0.0
        for child in child_nodes:
            contributions.extend(self._collect_contributions(child))
            child_new += child.new_count
            child_learn += child.learn_count
            child_review += child.review_count
            child_interday += float(
                getattr(child, "interday_learning", getattr(child, "interday_learning_uncapped", 0.0))
            )

        direct_new = max(deck_node.new_count - child_new, 0)
        direct_learn = max(deck_node.learn_count - child_learn, 0)
        direct_review = max(deck_node.review_count - child_review, 0)
        node_interday = float(
            getattr(deck_node, "interday_learning", getattr(deck_node, "interday_learning_uncapped", 0.0))
        )
        direct_interday = max(node_interday - child_interday, 0.0)

        if direct_new or direct_learn or direct_review or direct_interday:
            config_proto, config_dict = self._fetch_config(deck_id, deck_name)
            cutoff_ms = self._parse_ignore_before(config_dict.get("ignoreRevlogsBeforeDate", ""))
            averages = self._average_durations(deck_id, cutoff_ms)
            self._log(
                f"[deck] {deck_name} direct averages new={averages.new_seconds:.2f}s "
                f"review={averages.review_seconds:.2f}s cutoff={cutoff_ms}"
            )

            direct_total = direct_new + direct_learn + direct_review
            direct_node = SimpleNamespace(
                new_count=direct_new,
                learn_count=direct_learn,
                review_count=direct_review,
                interday_learning_uncapped=direct_interday,
                total_including_children=direct_total,
                total_in_deck=direct_total,
                children=[],
                name=getattr(deck_node, "name", ""),
            )

            exposures = self._fsrs_exposures(
                deck_id,
                deck,
                direct_node,
                config_proto,
                config_dict,
            )

            new_exposure_per_card = (
                exposures.new / direct_new if direct_new > 0 else 0.0
            )
            review_exposure_per_card = (
                exposures.review / direct_review if direct_review > 0 else 0.0
            )

            contributions.append(
                Contribution(
                    name=deck_name,
                    new_capacity=direct_new,
                    learn_count=direct_learn,
                    interday_count=direct_interday,
                    review_capacity=direct_review,
                    new_exposure_per_card=new_exposure_per_card,
                    review_exposure_per_card=review_exposure_per_card,
                    avg_new_seconds=averages.new_seconds,
                    avg_review_seconds=averages.review_seconds,
                )
            )

        return contributions

    def _aggregate_contributions(
        self, deck_node, deck_name: str, contributions: List[Contribution]
    ) -> tuple[float, float]:
        if not contributions:
            return 0.0, 0.0

        new_total = deck_node.new_count
        learn_total = deck_node.learn_count
        review_total = deck_node.review_count
        interday_total = float(
            getattr(deck_node, "interday_learning", getattr(deck_node, "interday_learning_uncapped", 0.0))
        )

        total_new_capacity = sum(c.new_capacity for c in contributions)
        total_learn = sum(c.learn_count for c in contributions)
        total_interday = sum(c.interday_count for c in contributions)
        total_review_capacity = sum(c.review_capacity for c in contributions)

        if self.debug:
            self._log(
                f"[deck] {deck_name} aggregate totals new={new_total} review={review_total} "
                f"learn={learn_total} interday={interday_total}"
            )

        new_ratio = new_total / total_new_capacity if total_new_capacity else 0.0
        review_ratio = review_total / total_review_capacity if total_review_capacity else 0.0
        learn_ratio = learn_total / total_learn if total_learn else 0.0
        interday_ratio = (
            interday_total / total_interday if total_interday else 0.0
        )

        new_ratio = min(new_ratio, 1.0) if new_ratio else 0.0
        review_ratio = min(review_ratio, 1.0) if review_ratio else 0.0
        learn_ratio = min(learn_ratio, 1.0) if learn_ratio else 0.0
        interday_ratio = min(interday_ratio, 1.0) if interday_ratio else 0.0

        new_time = 0.0
        learn_time = 0.0
        review_time = 0.0

        for contrib in contributions:
            new_alloc = contrib.new_capacity * new_ratio
            review_alloc = contrib.review_capacity * review_ratio
            learn_alloc = contrib.learn_count * learn_ratio
            interday_alloc = contrib.interday_count * interday_ratio

            new_time += new_alloc * contrib.new_exposure_per_card * contrib.avg_new_seconds
            learn_time += (learn_alloc + interday_alloc) * contrib.avg_new_seconds
            review_time += (
                review_alloc
                * contrib.review_exposure_per_card
                * contrib.avg_review_seconds
            )

            if self.debug:
                self._log(
                    f"[deck] {deck_name} contribution {contrib.name}: "
                    f"alloc new={new_alloc:.2f} review={review_alloc:.2f} "
                    f"learn={learn_alloc:.2f} interday={interday_alloc:.2f}"
                )

        total = new_time + learn_time + review_time
        if self.debug:
            self._log(
                f"[deck] {deck_name} aggregate time new={new_time:.2f}s "
                f"learn={learn_time:.2f}s review={review_time:.2f}s"
            )
        return total, review_time

    def _fetch_config(self, deck_id: DeckId, deck_name: str):
        try:
            config_proto = self.col._backend.get_deck_config(deck_id)
        except Exception as err:
            self._log(f"[deck] {deck_name} missing backend config: {err}")
            config_proto = None
        try:
            config_dict = self.col.decks.config_dict_for_deck_id(deck_id)
        except Exception as err:
            self._log(f"[deck] {deck_name} missing deck config: {err}")
            config_dict = {}
        return config_proto, config_dict

    # ------------------------------------------------------------------
    # FSRS exposures
    # ------------------------------------------------------------------
    def _fsrs_exposures(
        self,
        deck_id: DeckId,
        deck: dict,
        deck_node,
        config_proto,
        config_dict,
    ) -> FsrsExposureCounts:
        if not isinstance(deck, dict):
            deck = {}
        if not isinstance(config_dict, dict):
            config_dict = {}

        deck_name = self._deck_name(deck_node, deck_id)

        raw_new = config_dict.get("new")
        new_conf = raw_new.copy() if isinstance(raw_new, dict) else {}
        raw_rev = config_dict.get("rev")
        rev_conf = raw_rev.copy() if isinstance(raw_rev, dict) else {}
        raw_lapse = config_dict.get("lapse")
        lapse_conf = raw_lapse.copy() if isinstance(raw_lapse, dict) else {}

        new_limit = max(int(new_conf.get("perDay", 0)), 0)
        review_limit = max(int(rev_conf.get("perDay", 0)), 0)
        new_due = deck_node.new_count if new_limit == 0 else min(deck_node.new_count, new_limit)
        review_due = (
            deck_node.review_count if review_limit == 0 else min(deck_node.review_count, review_limit)
        )
        interday_learning = float(getattr(deck_node, "interday_learning_uncapped", 0))

        self._log(
            f"[deck] {deck_name} limits new={new_limit} review={review_limit} "
            f"due/new={deck_node.new_count}->{new_due} review={deck_node.review_count}->{review_due} "
            f"interday_learning={interday_learning}"
        )

        sim_new = 0.0
        sim_review = 0.0
        params = self._fsrs_params(config_proto, config_dict)
        if self.debug:
            self._log(
                f"[deck] {deck_name} using FSRS params length={len(params)}"
            )
        if params:
            try:
                resp = self.col._backend.simulate_fsrs_review(
                    SimulateFsrsReviewRequest(
                        params=params,
                        desired_retention=float(config_dict.get("desiredRetention", 0.9)),
                        deck_size=int(
                            getattr(deck_node, "total_including_children", 0)
                            or getattr(deck_node, "total_in_deck", 0)
                        ),
                        days_to_simulate=1,
                        new_limit=new_limit,
                        review_limit=review_limit,
                        max_interval=int(rev_conf.get("maxIvl", 0)),
                        search=self._deck_search(deck.get("name", deck_node.name)),
                        new_cards_ignore_review_limit=bool(
                            new_conf.get("ignoreReviewLimit", False)
                        ),
                        easy_days_percentages=_ensure_list(
                            config_dict.get("easyDaysPercentages")
                        ),
                        review_order=int(config_dict.get("reviewOrder", 0)),
                        suspend_after_lapse_count=int(lapse_conf.get("leechFails", 0)),
                    )
                )
                if resp.daily_new_count:
                    sim_new = float(resp.daily_new_count[0])
                if resp.daily_review_count:
                    sim_review = float(resp.daily_review_count[0])
            except Exception:
                # Fail quietly and use fallbacks.
                sim_new = 0.0
                sim_review = 0.0

        self._log(
            f"[deck] {deck_name} simulation new={sim_new} review={sim_review}"
        )

        exposures_per_new = self._new_exposures_per_card(sim_new, new_limit, new_conf)
        exposures_per_review = self._review_exposures_per_card(sim_review, review_limit)

        new_exposures = new_due * exposures_per_new
        review_exposures = review_due * exposures_per_review

        if new_exposures <= 0 and (deck_node.new_count or interday_learning):
            new_exposures = float(deck_node.new_count + interday_learning)
        if review_exposures <= 0 and deck_node.review_count:
            review_exposures = float(deck_node.review_count)

        self._log(
            f"[deck] {deck_name} exposures new={new_exposures:.2f} review={review_exposures:.2f}"
        )

        return FsrsExposureCounts(new=new_exposures, review=review_exposures)

    def _time_for_node(
        self,
        deck_id: DeckId,
        deck: dict,
        deck_node,
        config_proto,
        config_dict,
        averages: AverageDurations,
        deck_name: str,
    ) -> float:
        exposures = self._fsrs_exposures(
            deck_id, deck, deck_node, config_proto, config_dict
        )
        learning_exposures = deck_node.learn_count + float(
            getattr(deck_node, "interday_learning_uncapped", 0)
        )
        new_time = (exposures.new + learning_exposures) * averages.new_seconds
        review_time = exposures.review * averages.review_seconds
        total = new_time + review_time
        self._log(
            f"[deck] {deck_name} node time new={new_time:.2f}s review={review_time:.2f}s "
            f"(learning={learning_exposures:.2f}) -> total={total:.2f}s"
        )
        return total

    def _new_exposures_per_card(
        self, sim_new: float, new_limit: int, new_conf: dict
    ) -> float:
        if new_limit > 0 and sim_new > 0:
            ratio = sim_new / new_limit
            if ratio > 0:
                return ratio
        delays = new_conf.get("delays", [])
        steps = len(delays)
        return float(max(1, steps + 1))

    @staticmethod
    def _review_exposures_per_card(sim_review: float, review_limit: int) -> float:
        if review_limit > 0 and sim_review > 0:
            ratio = sim_review / review_limit
            if ratio >= 1:
                return ratio
        return 1.0

    @staticmethod
    def _fsrs_params(config_proto, config_dict: dict | None) -> Sequence[float]:
        # Prefer deck-specific parameters in descending order of FSRS version.
        if isinstance(config_dict, dict):
            for key in ("fsrsParams6", "fsrsParams5", "fsrsWeights"):
                params = config_dict.get(key)
                if params:
                    return [float(x) for x in params]
        if config_proto is not None:
            proto = config_proto.config
            if proto.fsrs_params_6:
                return list(proto.fsrs_params_6)
            if proto.fsrs_params_5:
                return list(proto.fsrs_params_5)
            if proto.fsrs_params_4:
                return list(proto.fsrs_params_4)
        return DEFAULT_FSRS_PARAMS

    @staticmethod
    def _deck_search(name: str) -> str:
        escaped = name.replace("\"", "\\\"")
        return f'deck:"{escaped}"'

    # ------------------------------------------------------------------
    # Average durations
    # ------------------------------------------------------------------
    def _average_durations(self, deck_id: DeckId, cutoff_ms: int | None) -> AverageDurations:
        cache_entry = self._avg_cache.get(int(deck_id))
        if cache_entry and cache_entry[0] == cutoff_ms:
            return cache_entry[1]

        deck_ids = [int(did) for did in self.col.decks.deck_and_child_ids(deck_id)]
        avg_new = self._avg_time_for(deck_ids, (0,), cutoff_ms)
        avg_review = self._avg_time_for(deck_ids, (1, 2), cutoff_ms)

        if avg_new is None:
            avg_new = self._global_average((0,), cutoff_ms)
        if avg_new is None:
            avg_new = FALLBACK_NEW_SECONDS

        if avg_review is None:
            avg_review = self._global_average((1, 2), cutoff_ms)
        if avg_review is None:
            avg_review = FALLBACK_REVIEW_SECONDS

        payload = AverageDurations(new_seconds=avg_new, review_seconds=avg_review)
        self._avg_cache[int(deck_id)] = (cutoff_ms, payload)
        return payload

    def _avg_time_for(
        self,
        deck_ids: Sequence[int],
        types: Sequence[int],
        cutoff_ms: int | None,
    ) -> float | None:
        if not deck_ids:
            return None
        placeholders = ",".join("?" for _ in deck_ids)
        type_clause = ",".join(str(t) for t in types)
        sql = (
            "SELECT AVG(revlog.time) FROM revlog "
            "JOIN cards ON cards.id = revlog.cid "
            f"WHERE cards.did IN ({placeholders}) "
            f"AND revlog.type IN ({type_clause})"
        )
        args: List[Any] = list(deck_ids)
        if cutoff_ms is not None:
            sql += " AND revlog.id >= ?"
            args.append(int(cutoff_ms))
        value = self.col.db.scalar(sql, *args)
        if value:
            return float(value) / 1000.0
        return None

    def _global_average(self, types: Sequence[int], cutoff_ms: int | None) -> float | None:
        key = (cutoff_ms or 0, ",".join(str(t) for t in types))
        if key in self._global_avg_cache:
            return self._global_avg_cache[key]
        type_clause = ",".join(str(t) for t in types)
        sql = f"SELECT AVG(time) FROM revlog WHERE type IN ({type_clause})"
        args: List[Any] = []
        if cutoff_ms is not None:
            sql += " AND id >= ?"
            args.append(int(cutoff_ms))
        value = self.col.db.scalar(sql, *args)
        result = float(value) / 1000.0 if value else None
        if result is not None:
            self._global_avg_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ignore_before(raw: str | None) -> int | None:
        if not raw:
            return None
        try:
            dt = datetime.strptime(raw.strip(), "%Y-%m-%d")
        except ValueError:
            return None

        epoch = datetime(1970, 1, 1)
        delta = dt - epoch
        millis = int(delta.total_seconds() * 1000)
        if millis < 0:
            return 0
        return millis


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if minutes:
            return f"{hours}h {minutes:02d}m"
        return f"{hours}h"
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining = int(round(seconds % 60))
        if remaining:
            return f"{minutes}m {remaining}s"
        return f"{minutes}m"
    return f"{int(round(seconds))}s"
