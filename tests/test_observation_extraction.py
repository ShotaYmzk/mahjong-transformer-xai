from pathlib import Path

import numpy as np

from data.observation_schema import (
    EVENT_TYPES,
    build_dataset_rows_from_xml,
    rows_to_npz_dict,
    validate_no_private_leakage,
)
from scripts.export_decision_report import extract_round_examples
from utils.xml_parser import parse_tenhou_xml


def test_tenhou_extraction_has_no_opponent_hand_leak(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        '<mjloggm ver="2.3">'
        '<GO type="169" lobby="0"/>'
        '<UN n0="a" n1="b" n2="c" n3="d" dan="1,1,1,1" rate="1500,1500,1500,1500" sx="M,M,M,M"/>'
        '<TAIKYOKU oya="0"/>'
        '<INIT seed="0,0,0,0,0,0" ten="250,250,250,250" oya="0" '
        'hai0="0,4,8,12,16,20,24,28,32,36,40,44,48" '
        'hai1="52,56,60,64,68,72,76,80,84,88,92,96,100" '
        'hai2="104,108,112,116,120,124,128,132,1,5,9,13,17" '
        'hai3="21,25,29,33,37,41,45,49,53,57,61,65,69"/>'
        '<T72/><D72/>'
        '<U73/><E73/>'
        '<RYUUKYOKU ba="0,0" sc="250,0,250,0,250,0,250,0"/>'
        '</mjloggm>',
        encoding="utf-8",
    )

    rows, report = build_dataset_rows_from_xml(xml)
    assert report.samples == 2
    data = rows_to_npz_dict(rows)
    assert validate_no_private_leakage(data) == []
    assert "valid_masks" in data
    assert np.all(data["valid_masks"][np.arange(len(data["labels"])), data["labels"]] == 1.0)
    for key in data:
        assert "opponent" not in key.lower()
        assert "hands_all" not in key.lower()

    p1_row = rows[1]
    non_padding = p1_row.sequence_features[p1_row.sequence_features[:, 0] != EVENT_TYPES["PADDING"]]
    draw_events = non_padding[non_padding[:, 0] == EVENT_TYPES["DRAW"]]
    assert draw_events.shape[0] == 1
    assert draw_events[0, 1] == 2.0  # P1's own draw only; P0's previous private draw is omitted.
    assert draw_events[0, 2] > 0.0
    assert not np.any((data["sequence_features"][:, :, 0] == EVENT_TYPES["DRAW"]) & (data["sequence_features"][:, :, 2] == 0))


def test_report_sequence_omits_opponent_private_draws(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        '<mjloggm ver="2.3">'
        '<GO type="169" lobby="0"/>'
        '<UN n0="a" n1="b" n2="c" n3="d" dan="1,1,1,1" rate="1500,1500,1500,1500" sx="M,M,M,M"/>'
        '<TAIKYOKU oya="0"/>'
        '<INIT seed="0,0,0,0,0,0" ten="250,250,250,250" oya="0" '
        'hai0="0,4,8,12,16,20,24,28,32,36,40,44,48" '
        'hai1="52,56,60,64,68,72,76,80,84,88,92,96,100" '
        'hai2="104,108,112,116,120,124,128,132,1,5,9,13,17" '
        'hai3="21,25,29,33,37,41,45,49,53,57,61,65,69"/>'
        '<T72/><D72/>'
        '<U73/><E73/>'
        '<RYUUKYOKU ba="0,0" sc="250,0,250,0,250,0,250,0"/>'
        '</mjloggm>',
        encoding="utf-8",
    )

    meta, rounds = parse_tenhou_xml(str(xml))
    examples = extract_round_examples(
        xml,
        meta,
        rounds[0],
        include_call_discards=True,
        remaining=10,
        summary={"skipped_invalid_label": 0},
    )
    events = [event for event in examples[1]["board"]["sequence_events"] if not event["is_padding"]]
    draw_events = [event for event in events if event["type"] == "DRAW"]
    assert len(draw_events) == 1
    assert draw_events[0]["player"] == 1
    assert draw_events[0]["tile"] is not None
    assert all(not event["hidden"] for event in events)
