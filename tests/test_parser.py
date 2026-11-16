import hashlib

import pytest

from src.parser import OsuParseError, extract_features, parse_osu_bytes, parse_osu_file

MINIMAL = b"""osu file format v14

[General]
Mode: 0

[Metadata]
Title:Test Song
Artist:Test Artist
Creator:tester
Version:Hard
Tags:j-rock energetic
BeatmapID:123
BeatmapSetID:45

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:8
ApproachRate:9
SliderMultiplier:1.8
SliderTickRate:1

[TimingPoints]
0,333.333333,4,2,1,60,1,0
2000,-50,4,2,1,60,0,0

[HitObjects]
256,192,0,5,0,0:0:0:0:
100,100,166,1,0,0:0:0:0:
400,300,333,1,0,0:0:0:0:
256,192,1000,2,0,L|356:192,1,180
256,192,3000,12,0,4000,0:0:0:0:
"""


def test_metadata_and_difficulty():
    bm = parse_osu_bytes(MINIMAL)
    assert bm.title == "Test Song" and bm.artist == "Test Artist" and bm.version == "Hard"
    assert bm.tags == ["j-rock", "energetic"]
    assert (bm.beatmap_id, bm.beatmapset_id) == (123, 45)
    assert (bm.hp, bm.cs, bm.od, bm.ar) == (5, 4, 8, 9)
    assert bm.md5 == hashlib.md5(MINIMAL).hexdigest()


def test_hit_objects_types_and_slider_duration():
    bm = parse_osu_bytes(MINIMAL)
    assert len(bm.hit_objects) == 5
    circle, *_, slider, spinner = bm.hit_objects
    assert circle.is_circle and circle.new_combo
    assert slider.is_slider and not slider.is_circle
    # 180px / (1.8 * 100 * SV1) = 1 beat = 333ms
    assert slider.end_time == pytest.approx(1000 + 333, abs=1)
    assert spinner.is_spinner and spinner.end_time == 4000


def test_inherited_timing_point_sv():
    bm = parse_osu_bytes(MINIMAL)
    green = bm.timing_points[1]
    assert not green.uninherited and green.sv_multiplier == pytest.approx(2.0)
    assert bm.timing_points[0].bpm == pytest.approx(180, abs=0.01)


def test_ar_defaults_to_od_on_old_maps():
    raw = MINIMAL.replace(b"ApproachRate:9\n", b"")
    assert parse_osu_bytes(raw).ar == 8


def test_invalid_header_raises():
    with pytest.raises(OsuParseError):
        parse_osu_bytes(b"not an osu file")


def test_all_samples_parse(sample_files):
    for path in sample_files:
        bm = parse_osu_file(path)
        assert bm.mode == 0 and len(bm.hit_objects) > 50, path.name
        assert len(bm.md5) == 32


def _features(sample_files, title):
    path = next(p for p in sample_files if f"- {title} (" in p.name)
    return extract_features(parse_osu_file(path))


def test_feature_extractor_distinguishes_profiles(sample_files):
    stream = _features(sample_files, "Hyperion Gate")
    jumps = _features(sample_files, "Blazing Requiem")
    tech = _features(sample_files, "Static Bloom")
    calm = _features(sample_files, "Lo-fi Lantern")

    assert stream.max_stream_len >= 32 and "long_streams" in stream.skills
    assert stream.speed_score > jumps.speed_score
    assert jumps.aim_score > 0.8 and "jumps" in jumps.skills and jumps.max_stream_len < 7
    assert tech.tech_score > jumps.tech_score and "tech" in tech.skills
    assert tech.sv_change_count >= 3
    assert calm.star_rating < 3.5 < jumps.star_rating
    assert calm.bpm == pytest.approx(90, abs=0.5)


def test_features_are_deterministic(sample_files):
    a = extract_features(parse_osu_file(sample_files[0]))
    b = extract_features(parse_osu_file(sample_files[0]))
    assert a == b


def _circles_map(gaps_in_beats, bpm=150, circular=False):
    import math

    beat = 60000 / bpm
    t, lines = 1000.0, []
    for i, g in enumerate(gaps_in_beats):
        if circular:  # octógono de raio 150: ângulos de 135° e ~115px entre notas (movimento "flow")
            x, y = int(256 + 150 * math.cos(i * math.pi / 4)), int(192 + 150 * math.sin(i * math.pi / 4))
        else:
            x, y = 100 + (i % 2) * 200, 192
        lines.append(f"{x},{y},{int(t)},1,0,0:0:0:0:")
        t += g * beat
    head = MINIMAL.split(b"[TimingPoints]")[0].decode()
    return parse_osu_bytes((head + f"[TimingPoints]\n0,{beat},4,2,1,60,1,0\n\n[HitObjects]\n"
                            + "\n".join(lines) + "\n").encode())


def test_rhythm_variety_measures_mixing_not_odd_snaps():
    """Um mapa inteiro em 1/3 é consistente; tech de ritmo é MISTURAR snaps."""
    triplets = extract_features(_circles_map([1 / 3] * 120))
    mixed = extract_features(_circles_map([1 / 2, 1 / 3, 3 / 4, 1 / 2, 1, 2 / 3] * 20))
    assert triplets.rhythm_variety < 0.05
    assert mixed.rhythm_variety > 0.3 > triplets.rhythm_variety


def test_slow_wide_maps_are_not_flow_aim():
    """Easy/Normal com spacing grande e ângulos abertos, mas cursor lento, não é flow aim."""
    slow = extract_features(_circles_map([2.0] * 80, bpm=90, circular=True))
    fast = extract_features(_circles_map([0.5] * 160, bpm=180, circular=True))
    assert slow.obtuse_angle_ratio > 0.9 and fast.obtuse_angle_ratio > 0.9  # mesma geometria
    assert "flow_aim" not in slow.skills and slow.flow_score < 0.2
    assert "flow_aim" in fast.skills
