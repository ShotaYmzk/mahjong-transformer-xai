from pathlib import Path

import numpy as np

from data.observation_schema import (
    build_dataset_rows_from_xml,
    rows_to_npz_dict,
    validate_no_private_leakage,
)


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
