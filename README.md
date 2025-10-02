# FSRS Time Estimate Add-on

This add-on extends the Anki deck browser and overview screens with
FSRS-based study time estimates:

- **Total workload** combines the expected time for all new, learning,
  interday learning, and review cards that can appear today.
- **Review-only workload** shows the subset for scheduled reviews.
- Estimates respect each deck's FSRS preset, deck options (including
  "ignore reviews before"), and the actual counts Anki places in the
  parent's queues.
- Optional debug logging (toggle via add-on config or
  `FSRS_TIME_DEBUG=1`) inserts an expandable log detailing the
  individual contributions and queue breakdowns.

## Development

- Python 3.10+
- Requires the Anki Python libraries (`anki`, `aqt`).

### Config

Add `config.json` with:

```json
{ "debug": true }
```

or start Anki with `FSRS_TIME_DEBUG=1` to see detailed logs for each
screen.

### Files

- `estimator.py` – core estimator logic.
- `__init__.py` – Anki hooks and UI integration.

## License

Released under the GNU Affero General Public License v3. See
[LICENSE](LICENSE).
