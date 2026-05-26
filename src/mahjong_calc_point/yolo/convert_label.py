try:
    from mahjong_calc_point.tiles import TILE_LABELS
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mahjong_calc_point.tiles import TILE_LABELS


def convert(idx: int) -> str:
    return TILE_LABELS[idx]
