from flask import Flask, jsonify, render_template, request
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.tile import TilesConverter
from enum import Enum
from typing import Any, Dict

from mahjong_calc_point.detection import detect_tiles

app = Flask(__name__)
calculator = HandCalculator()


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
    }
    return (
        tiles_dict,
        win_tile_dict,
        melds_dict,
        dora_indicators_dict,
        config_dict,
    )


def serialize_calculation_result(result: Any) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "yaku": [str(yaku) for yaku in result.yaku],
            "han": result.han,
            "fu": result.fu,
            "cost": result.cost.get("main", ""),
        }
    except Exception:
        return {
            "ok": False,
            "error": str(result),
            "yaku": str(result),
            "han": "",
            "fu": "",
            "cost": "",
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
                    melds.append(
                        Meld(meld_type=meld_kind, tiles=meld_tile, opened=is_minkan)
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
                yaku = result.yaku
                han = result.han
                fu = result.fu
                cost = result.cost["main"]
                # print_hand_result(result)
            except Exception as e:
                print(result)
                print(f"An error occurred: {str(e)}")
                yaku = result
                han = ""
                fu = ""
                cost = ""

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
    return jsonify(serialize_calculation_result(result))


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
