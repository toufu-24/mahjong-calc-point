from flask import Flask, render_template, request
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.tile import TilesConverter
from enum import Enum
from typing import Dict

app = Flask(__name__)
calculator = HandCalculator()


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
        # 手牌
        tiles_dict = {
            "man": request.form.get("man", ""),
            "pin": request.form.get("pin", ""),
            "sou": request.form.get("sou", ""),
            "honors": request.form.get("honors", ""),
        }
        # 和了牌
        win_tile_dict = {
            "man": request.form.get("win_man", ""),
            "pin": request.form.get("win_pin", ""),
            "sou": request.form.get("win_sou", ""),
            "honors": request.form.get("win_honors", ""),
        }
        # configの情報を取得
        config_dict = {
            "is_riichi": request.form.get("is_riichi", "off") == "on",
            "is_daburu_riichi": request.form.get("is_daburu_riichi", "off") == "on",
            "is_tsumo": request.form.get("is_tsumo", "off") == "on",
            "is_ippatsu": request.form.get("is_ippatsu", "off") == "on",
            "is_chankan": request.form.get("is_chankan", "off") == "on",
            "is_rinshan": request.form.get("is_rinshan", "off") == "on",
            "is_haitei": request.form.get("is_haitei", "off") == "on",
            "is_houtei": request.form.get("is_houtei", "off") == "on",
            "is_nagashi_mangan": request.form.get("is_nagashi_mangan", "off") == "on",
            "is_tenhou": request.form.get("is_tenhou", "off") == "on",
            "is_chiihou": request.form.get("is_chiihou", "off") == "on",
            "is_renhou": request.form.get("is_renhou", "off") == "on",
        }

        # 副露の情報を各牌種別ごとに取得
        melds_dict = {
            "man": request.form.get("melds_man", ""),
            "pin": request.form.get("melds_pin", ""),
            "sou": request.form.get("melds_sou", ""),
            "honors": request.form.get("melds_honors", ""),
        }

        # ドラ表示牌の情報を各牌種別ごとに取得
        dora_indicators_dict = {
            "man": request.form.get("dora_man", ""),
            "pin": request.form.get("dora_pin", ""),
            "sou": request.form.get("dora_sou", ""),
            "honors": request.form.get("dora_honors", ""),
        }

        # 手牌計算の関数呼び出し
        result = calculate_hand(
            tiles_dict, win_tile_dict, melds_dict, dora_indicators_dict, config_dict
        )
        print(HandConstants.KAZOE_LIMITED)
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


if __name__ == "__main__":
    app.run(debug=True)
