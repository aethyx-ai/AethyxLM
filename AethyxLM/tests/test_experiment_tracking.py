import json

from tracking import JsonlExperimentTracker


def test_jsonl_tracker_writes_run_and_metric_events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tracker = JsonlExperimentTracker(path, run_id="run-1", metadata={"model": "test"})
    tracker.log("train_metrics", step=10, loss=2.5)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["run_started", "train_metrics"]
    assert records[1]["run_id"] == "run-1"
    assert records[1]["step"] == 10
    assert records[1]["loss"] == 2.5

