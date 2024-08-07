from flask import Flask, render_template, request
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.tile import TilesConverter

app = Flask(__name__)
calculator = HandCalculator()

# useful helper
def print_hand_result(hand_result):
    print(hand_result.han, hand_result.fu)
    print(hand_result.cost['main'])
    print(hand_result.yaku)
    for fu_item in hand_result.fu_details:
        print(fu_item)
    print('')


def calculate_hand(tiles_str, win_tile_str, is_tsumo=False, melds_str=None, dora_indicators_str=None):
    try:
        tiles = TilesConverter.string_to_136_array(
            man=tiles_str.get('man', ''),
            pin=tiles_str.get('pin', ''),
            sou=tiles_str.get('sou', '')
        )
        win_tile = TilesConverter.string_to_136_array(
            man=win_tile_str.get('man', ''),
            pin=win_tile_str.get('pin', ''),
            sou=win_tile_str.get('sou', '')
        )[0]
    except IndexError:
        return "Invalid input for tiles or win tile"

    melds = []
    if melds_str:
        try:
            for meld_str in melds_str.split(','):
                parts = meld_str.split(':')
                if len(parts) == 3:
                    meld = TilesConverter.string_to_136_array(man=parts[0], pin=parts[1], sou=parts[2])
                    melds.append(Meld(meld_type=Meld.PON, tiles=meld))
                else:
                    return "Invalid input for melds"
        except IndexError:
            return "Invalid input for melds"

    dora_indicators = []
    if dora_indicators_str:
        try:
            for part in dora_indicators_str.split(','):
                kind, tile = part.split(':')
                if kind == 'man':
                    dora_tile = TilesConverter.string_to_136_array(man=tile)[0]
                elif kind == 'pin':
                    dora_tile = TilesConverter.string_to_136_array(pin=tile)[0]
                elif kind == 'sou':
                    dora_tile = TilesConverter.string_to_136_array(sou=tile)[0]
                else:
                    return "Invalid input for dora indicators"
                dora_indicators.append(dora_tile)
        except (IndexError, ValueError):
            return "Invalid input for dora indicators"

    config = HandConfig(is_tsumo=is_tsumo)
    try:
        result = calculator.estimate_hand_value(tiles, win_tile, melds=melds, dora_indicators=dora_indicators, config=config)
    except Exception as e:
        return f"An error occurred: {str(e)}"

    print_hand_result(result)
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    yaku = None
    han = None
    fu = None
    cost = None
    if request.method == 'POST':
        tiles_str = {
            'man': request.form.get('man', ''),
            'pin': request.form.get('pin', ''),
            'sou': request.form.get('sou', '')
        }
        win_tile_str = {
            'man': request.form.get('win_man', ''),
            'pin': request.form.get('win_pin', ''),
            'sou': request.form.get('win_sou', '')
        }
        is_tsumo = request.form.get('is_tsumo', 'off') == 'on'
        
        melds_str = request.form.get('melds', '')
        dora_indicators_str = request.form.get('dora_indicators', '')
        
        result = calculate_hand(tiles_str, win_tile_str, is_tsumo, melds_str, dora_indicators_str)
        
        if result:
            print('yaku:', result.yaku)
            yaku = result.yaku
            han = result.han
            fu = result.fu
            cost = result.cost['main']
            print(result.cost['main'])

    return render_template('index.html', result=result, tiles_str=tiles_str, win_tile_str=win_tile_str, melds_str=melds_str, dora_indicators_str=dora_indicators_str, yaku=yaku, han=han, fu=fu, cost=cost)


if __name__ == '__main__':
    app.run(debug=True)
