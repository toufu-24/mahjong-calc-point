from flask import Flask, jsonify, render_template, request
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.tile import TilesConverter
from mahjong.constants import EAST, SOUTH
from enum import Enum
from typing import Any, Dict, Optional

from mahjong_calc_point.detection import detect_tiles

app = Flask(__name__)
calculator = HandCalculator()

YAKU_JAPANESE_NAMES = {
    0: "門前清自摸和",
    1: "立直",
    2: "オープン立直",
    3: "一発",
    4: "槍槓",
    5: "嶺上開花",
    6: "海底摸月",
    7: "河底撈魚",
    8: "ダブル立直",
    9: "ダブルオープン立直",
    10: "流し満貫",
    11: "人和",
    12: "平和",
    13: "断么九",
    14: "一盃口",
    15: "役牌 白",
    16: "役牌 発",
    17: "役牌 中",
    18: "自風 東",
    19: "自風 南",
    20: "自風 西",
    21: "自風 北",
    22: "場風 東",
    23: "場風 南",
    24: "場風 西",
    25: "場風 北",
    26: "三色同順",
    27: "一気通貫",
    28: "混全帯么九",
    29: "混老頭",
    30: "対々和",
    31: "三暗刻",
    32: "三槓子",
    33: "三色同刻",
    34: "七対子",
    35: "小三元",
    36: "混一色",
    37: "純全帯么九",
    38: "二盃口",
    39: "清一色",
    100: "国士無双",
    101: "九蓮宝燈",
    102: "四暗刻",
    103: "大三元",
    104: "小四喜",
    105: "緑一色",
    106: "四槓子",
    107: "字一色",
    108: "清老頭",
    109: "大車輪",
    110: "大七星",
    111: "大四喜",
    112: "国士無双十三面待ち",
    113: "四暗刻単騎",
    114: "純正九蓮宝燈",
    115: "天和",
    116: "地和",
    117: "人和 役満",
    118: "責任払い",
    119: "八連荘",
    120: "ドラ",
    121: "赤ドラ",
    122: "裏ドラ",
}


def parse_calculation_form(form: Any):
    tiles_dict = {
        "man": form.get("man", ""),
        "pin": form.get("pin", ""),
        "sou": form.get("sou", ""),
        "honors": form.get("honors", ""),
    }
    win_tile_dict = {
        "man": form.get("win_man", ""),
        "pin": form.get("win_pin", ""),
        "sou": form.get("win_sou", ""),
        "honors": form.get("win_honors", ""),
    }
    melds_dict = {
        "man": form.get("melds_man", ""),
        "pin": form.get("melds_pin", ""),
        "sou": form.get("melds_sou", ""),
        "honors": form.get("melds_honors", ""),
    }
    dora_indicators_dict = {
        "man": form.get("dora_man", ""),
        "pin": form.get("dora_pin", ""),
        "sou": form.get("dora_sou", ""),
        "honors": form.get("dora_honors", ""),
    }
    config_dict = {
        "is_riichi": form.get("is_riichi", "off") == "on",
        "is_daburu_riichi": form.get("is_daburu_riichi", "off") == "on",
        "is_tsumo": form.get("is_tsumo", "off") == "on",
        "is_ippatsu": form.get("is_ippatsu", "off") == "on",
        "is_chankan": form.get("is_chankan", "off") == "on",
        "is_rinshan": form.get("is_rinshan", "off") == "on",
        "is_haitei": form.get("is_haitei", "off") == "on",
        "is_houtei": form.get("is_houtei", "off") == "on",
        "is_nagashi_mangan": form.get("is_nagashi_mangan", "off") == "on",
        "is_tenhou": form.get("is_tenhou", "off") == "on",
        "is_chiihou": form.get("is_chiihou", "off") == "on",
        "is_renhou": form.get("is_renhou", "off") == "on",
        "player_wind": EAST if form.get("is_dealer", "off") == "on" else SOUTH,
    }
    return (
        tiles_dict,
        win_tile_dict,
        melds_dict,
        dora_indicators_dict,
        config_dict,
    )


def format_payment(cost: dict[str, Any], is_tsumo: bool, is_dealer: bool) -> str:
    main = cost.get("main", "")
    additional = cost.get("additional", "")
    total = cost.get("total", "")

    if main == "":
        return ""

    if not is_tsumo:
        return f"放銃者から {main}点"

    if is_dealer:
        total_text = f" (合計 {total}点)" if total else ""
        return f"子3人から {main}点ずつ{total_text}"

    total_text = f" (合計 {total}点)" if total else ""
    return f"親から {main}点、子2人から {additional}点ずつ{total_text}"


def format_yaku_name(yaku: Any) -> str:
    yaku_id = getattr(yaku, "yaku_id", None)
    name = YAKU_JAPANESE_NAMES.get(yaku_id, str(yaku))
    if yaku_id in {120, 121, 122}:
        han = getattr(yaku, "han_closed", None) or getattr(yaku, "han_open", None)
        return f"{name} {han}" if han else name
    return name


def format_yaku_names(yaku_list: Any) -> list[str]:
    return [format_yaku_name(yaku) for yaku in yaku_list or []]


def serialize_calculation_result(
    result: Any, config_dict: Optional[Dict[str, Any]] = None
) -> dict[str, Any]:
    try:
        cost = result.cost
        config_dict = config_dict or {}
        is_tsumo = bool(config_dict.get("is_tsumo", False))
        is_dealer = config_dict.get("player_wind") == EAST
        return {
            "ok": True,
            "yaku": format_yaku_names(result.yaku),
            "han": result.han,
            "fu": result.fu,
            "cost": cost.get("main", ""),
            "payment": format_payment(cost, is_tsumo, is_dealer),
        }
    except Exception:
        return {
            "ok": False,
            "error": str(result),
            "yaku": str(result),
            "han": "",
            "fu": "",
            "cost": "",
            "payment": "",
        }


# useful helper
def print_hand_result(hand_result):
    try:
        print(hand_result.han, hand_result.fu)
        print(hand_result.cost["main"])
        print(hand_result.yaku)
        for fu_item in hand_result.fu_details:
            print(fu_item)
        print("")
    except Exception as e:
        print(f"An error occurred in print: {str(e)}")


def calculate_hand(
    tiles_dict: Dict[str, str],
    win_tile_dict: Dict[str, str],
    melds_dict: Dict[str, str],
    dora_indicators_dict: Dict[str, str],
    config_dict: Dict[str, bool],
):
    # 手牌計算
    try:
        tiles = TilesConverter.string_to_136_array(
            man=tiles_dict.get("man", ""),
            pin=tiles_dict.get("pin", ""),
            sou=tiles_dict.get("sou", ""),
            honors=tiles_dict.get("honors", ""),
        )
    except IndexError:
        return "Invalid input for tiles"
    try:
        win_tile = TilesConverter.string_to_136_array(
            man=win_tile_dict.get("man", ""),
            pin=win_tile_dict.get("pin", ""),
            sou=win_tile_dict.get("sou", ""),
            honors=win_tile_dict.get("honors", ""),
        )[0]
    except IndexError:
        return "Invalid input for win tile"

    # 副露の情報を取得
    # 鳴き(チー:CHI, ポン:PON, カン:KAN(True:ミンカン,False:アンカン), カカン:CHANKAN, ヌキドラ:NUKI)
    melds = []
    if melds_dict:
        try:
            for kind, meld_kind_str in melds_dict.items():
                # 副露の種類を判別
                meld_str = meld_kind_str.split(",")
                for meld in meld_str:
                    if meld == "":
                        continue
                    is_chi: bool = False
                    is_pon: bool = False
                    is_kan: bool = False
                    is_minkan: bool = False
                    if len(meld) == 4 + 1:
                        is_kan = True
                        if meld[0] != "a" and meld[0] != "m":
                            return "kan meld should start with 'a' or 'm'"
                        is_minkan = meld[0] == "m"
                        meld = meld[1:]
                    elif len(meld) == 3 + 1:
                        if meld[0] != "c" and meld[0] != "p":
                            return "chi or pon meld should start with 'c' or 'p'"
                        if meld[0] == "c":
                            is_chi = True
                        elif meld[0] == "p":
                            is_pon = True
                        meld = meld[1:]
                    else:
                        print(meld)
                        return "Invalid input for melds"
                    # 副露の牌を取得
                    if kind == "man":
                        meld_tile = TilesConverter.string_to_136_array(man=meld)
                    elif kind == "pin":
                        meld_tile = TilesConverter.string_to_136_array(pin=meld)
                    elif kind == "sou":
                        meld_tile = TilesConverter.string_to_136_array(sou=meld)
                    elif kind == "honors":
                        meld_tile = TilesConverter.string_to_136_array(honors=meld)
                    else:
                        return "Invalid input for melds"
                    # 副露の種類に応じてMeldKindを設定
                    if is_chi:
                        meld_kind = Meld.CHI
                    elif is_pon:
                        meld_kind = Meld.PON
                    elif is_kan:
                        meld_kind = Meld.KAN
                    else:
                        return "Invalid input for melds"
                    opened = True if is_chi or is_pon else is_minkan
                    melds.append(
                        Meld(meld_type=meld_kind, tiles=meld_tile, opened=opened)
                    )
        except (IndexError, ValueError):
            return "Invalid input for melds"

    dora_indicators = []
    if dora_indicators_dict:
        try:
            for kind, tile in dora_indicators_dict.items():
                for char in tile:
                    if char == "":
                        continue
                    if kind == "man":
                        dora_tile = TilesConverter.string_to_136_array(man=char)[0]
                    elif kind == "pin":
                        dora_tile = TilesConverter.string_to_136_array(pin=char)[0]
                    elif kind == "sou":
                        dora_tile = TilesConverter.string_to_136_array(sou=char)[0]
                    elif kind == "honors":
                        dora_tile = TilesConverter.string_to_136_array(honors=char)[0]
                    else:
                        return "Invalid input for dora indicators"
                    dora_indicators.append(dora_tile)
        except (IndexError, ValueError):
            return "Invalid input for dora indicators"

    config = HandConfig(**config_dict)
    try:
        result = calculator.estimate_hand_value(
            tiles, win_tile, melds=melds, dora_indicators=dora_indicators, config=config
        )
    except Exception as e:
        return f"An error occurred: {str(e)}"
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    yaku = ""
    han = ""
    fu = ""
    cost = ""
    payment = ""
    tiles_dict = {}
    win_tile_dict = {}
    melds_dict = {}
    dora_indicators_dict = {}

    if request.method == "POST":
        (
            tiles_dict,
            win_tile_dict,
            melds_dict,
            dora_indicators_dict,
            config_dict,
        ) = parse_calculation_form(request.form)

        # 手牌計算の関数呼び出し
        result = calculate_hand(
            tiles_dict, win_tile_dict, melds_dict, dora_indicators_dict, config_dict
        )
        # 結果を取得
        if result:
            try:
                yaku = ", ".join(format_yaku_names(result.yaku))
                han = result.han
                fu = result.fu
                cost = result.cost["main"]
                payment = format_payment(
                    result.cost,
                    config_dict.get("is_tsumo", False),
                    config_dict.get("player_wind") == EAST,
                )
                # print_hand_result(result)
            except Exception as e:
                print(result)
                print(f"An error occurred: {str(e)}")
                yaku = result
                han = ""
                fu = ""
                cost = ""
                payment = ""

    return render_template(
        "index.html",
        result=result,
        tiles_str=tiles_dict,
        win_tile_str=win_tile_dict,
        melds_str=melds_dict,
        dora_indicators_str=dora_indicators_dict,
        yaku=yaku,
        han=han,
        fu=fu,
        cost=cost,
        payment=payment,
    )


@app.route("/api/calculate", methods=["POST"])
def calculate():
    (
        tiles_dict,
        win_tile_dict,
        melds_dict,
        dora_indicators_dict,
        config_dict,
    ) = parse_calculation_form(request.form)
    result = calculate_hand(
        tiles_dict, win_tile_dict, melds_dict, dora_indicators_dict, config_dict
    )
    return jsonify(serialize_calculation_result(result, config_dict))


@app.route("/api/detect", methods=["POST"])
def detect():
    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"error": "image file is required"}), 400

    try:
        result = detect_tiles(image.read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"推論に失敗しました: {str(e)}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
