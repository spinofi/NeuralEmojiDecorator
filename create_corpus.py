# -*- coding: utf-8 -*-
import sys,json,gzip,re
import MeCab

parse = MeCab.Tagger("-Owakati").parse

emoji_list = [u'😀',u'😛',u'😸',u'👷',u'💋',u'😁',u'😜',u'😹',u'👸',u'👅',u'😂',u'😝',u'😺',u'💂',u'💅'
              ,u'😃',u'😞',u'😻',u'👼',u'👋',u'😄',u'😟',u'😼',u'🎅',u'👍',u'😅',u'😠',u'😽',u'👻',u'👎'
              ,u'😆',u'😡',u'😾',u'👹',u'😇',u'😢',u'😿',u'👺',u'👆',u'😈',u'😣',u'🙀',u'💩',u'👇',u'👿'
              ,u'😤',u'👣',u'💀',u'👈',u'😉',u'😥',u'👤',u'👽',u'👉',u'😊',u'😦',u'👥',u'👾',u'👌',u'😧'
              ,u'👦',u'🙇',u'😋',u'😨',u'👧',u'💁',u'👊',u'😌',u'😩',u'👨',u'🙅',u'✊',u'😍',u'😪',u'👩'
              ,u'🙆',u'✋',u'😎',u'😫',u'👪',u'🙋',u'💪',u'😏',u'😬',u'👫',u'🙎',u'👐',u'😐',u'😭',u'👬'
              ,u'🙍',u'🙏',u'😑',u'😮',u'👭',u'💆',u'😒',u'😯',u'👮',u'💇',u'😓',u'😰',u'👯',u'💑',u'😔'
              ,u'😱',u'👰',u'💏',u'😕',u'😲',u'👱',u'🙌',u'😖',u'😳',u'👲',u'👏',u'😗',u'😴',u'👳',u'👂'
              ,u'😘',u'😵',u'👴',u'👀',u'😙',u'😶',u'👵',u'👃',u'😚',u'😷',u'👶',u'👄',u'🌱',u'🌲',u'🌳'
              ,u'🌴',u'🌵',u'🌷',u'🌸',u'🌹',u'🌺',u'🌻',u'🌼',u'💐',u'🌾',u'🌿',u'🍀',u'🍁',u'🍂',u'🍃'
              ,u'🍄',u'🌰',u'🐀',u'🐁',u'🐭',u'🐹',u'🐂',u'🐃',u'🐄',u'🐮',u'🐅',u'🐆',u'🐯',u'🐇',u'🐰'
              ,u'🐈',u'🐱',u'🐎',u'🐴',u'🐏',u'🐑',u'🐐',u'🐓',u'🐔',u'🐤',u'🐣',u'🐥',u'🐦',u'🐧',u'🐘'
              ,u'🐪',u'🐫',u'🐗',u'🐖',u'🐷',u'🐽',u'🐕',u'🐩',u'🐶',u'🐺',u'🐻',u'🐨',u'🐼',u'🐵',u'🙈'
              ,u'🙉',u'🙊',u'🐒',u'🐉',u'🐲',u'🐊',u'🐍',u'🐢',u'🐸',u'🐋',u'🐳',u'🐬',u'🐙',u'🐟',u'🐠'
              ,u'🐡',u'🐚',u'🐌',u'🐛',u'🐜',u'🐝',u'🐞',u'🐾',u'⚡️',u'🔥',u'🌙',u'☀️',u'⛅',u'☁️',u'💧'
              ,u'💦',u'☔',u'💨',u'🌟',u'⭐',u'🌠',u'🌄',u'🌅',u'🌈',u'🌊',u'🌋',u'🌌',u'🗻',u'🗾',u'🌐'
              ,u'🌍',u'🌎',u'🌏',u'🌑',u'🌒',u'🌓',u'🌔',u'🌕',u'🌖',u'🌗',u'🌘',u'🌚',u'🌝',u'🌛',u'🌜'
              ,u'🌞',u'🍅',u'🍆',u'🌽',u'🍠',u'🍇',u'🍈',u'🍉',u'🍊',u'🍋',u'🍌',u'🍍',u'🍎',u'🍏',u'🍐'
              ,u'🍑',u'🍓',u'🍒',u'🍔',u'🍕',u'🍖',u'🍗',u'🍘',u'🍙',u'🍚',u'🍛',u'🍜',u'🍝',u'🍞',u'🍟'
              ,u'🍡',u'🍢',u'🍣',u'🍤',u'🍥',u'🍦',u'🍧',u'🍨',u'🍩',u'🍪',u'🍫',u'🍬',u'🍭',u'🍮',u'🍯'
              ,u'🍰',u'🍱',u'🍲',u'🍳',u'🍴',u'🍵',u'☕',u'🍶',u'🍷',u'🍸',u'🍹',u'🍺',u'🍻',u'🍼',u'🎂'
              ,u'🎃',u'🎄',u'🎋',u'🎍',u'🎑',u'🎆',u'🎇',u'🎉',u'🎊',u'🎈',u'💫',u'✨',u'💥',u'🎓',u'👑'
              ,u'🎎',u'🎏',u'🎐',u'🎌',u'🏮',u'💍',u'💔',u'💌',u'💕',u'💞',u'💓',u'💗',u'💖',u'💘',u'💝'
              ,u'💟',u'💜',u'💛',u'💚',u'💙',u'🏃',u'🚶',u'💃',u'🚣',u'🏊',u'🏄',u'🛀',u'🏂',u'🎿',u'⛄'
              ,u'🚴',u'🚵',u'🏇',u'⛺',u'🎣',u'⚽',u'🏀',u'🏈',u'⚾️',u'🎾',u'🏉',u'⛳',u'🏆',u'🎽',u'🏁'
              ,u'🎹',u'🎸',u'🎻',u'🎷',u'🎺',u'🎵',u'🎶',u'🎼',u'🎧',u'🎤',u'🎭',u'🎫',u'🎩',u'🎪',u'🎬'
              ,u'🎨',u'🎯',u'🎱',u'🎳',u'🎰',u'🎲',u'🎮',u'🎴',u'🃏',u'🀄',u'🎠',u'🎡',u'🎢',u'🚃',u'🚞'
              ,u'🚂',u'🚋',u'🚝',u'🚄',u'🚅',u'🚆',u'🚇',u'🚈',u'🚉',u'🚊',u'🚌',u'🚍',u'🚎',u'🚐',u'🚑'
              ,u'🚒',u'🚓',u'🚔',u'🚨',u'🚕',u'🚖',u'🚗',u'🚘',u'🚙',u'🚚',u'🚛',u'🚜',u'🚲',u'🚏',u'⛽'
              ,u'🚧',u'🚦',u'🚥',u'🚀',u'🚁',u'💺',u'⚓',u'🚢',u'🚤',u'⛵',u'🚡',u'🚠',u'🚟',u'🛂',u'🛃'
              ,u'🛄',u'🛅',u'💴',u'💶',u'💷',u'💵',u'🗽',u'🗿',u'🌁',u'🗼',u'⛲',u'🏰',u'🏯',u'🌇',u'🌆'
              ,u'🌃',u'🌉',u'🏠',u'🏡',u'🏢',u'🏬',u'🏭',u'🏣',u'🏤',u'🏥',u'🏨',u'🏩',u'💒',u'⛪',u'🏪'
              ,u'⌚',u'📱',u'📲',u'💻',u'⏰',u'⏳',u'⌛',u'📷',u'📹',u'🎥',u'📺',u'📻',u'📟',u'📞',u'📠'
              ,u'💽',u'💾',u'💿',u'📀',u'📼',u'🔋',u'🔌',u'💡',u'🔦',u'📡',u'💳',u'💸',u'💰',u'💎',u'🌂'
              ,u'👝',u'👛',u'👜',u'💼',u'🎒',u'💄',u'👓',u'👒',u'👡',u'👠',u'👢',u'👞',u'👟',u'👙',u'👗'
              ,u'👘',u'👚',u'👕',u'👔',u'👖',u'🚪',u'🚿',u'🛁',u'🚽',u'💈',u'💉',u'💊',u'🔬',u'🔭',u'🔮'
              ,u'🔧',u'🔪',u'🔩',u'🔨',u'💣',u'🚬',u'🔫',u'🔖',u'📰',u'🔑',u'📩',u'📨',u'📧',u'📥',u'📤'
              ,u'📦',u'📯',u'📮',u'📪',u'📫',u'📬',u'📭',u'📄',u'📃',u'📑',u'📈',u'📉',u'📊',u'📅',u'📆'
              ,u'🔅',u'🔆',u'📜',u'📋',u'📖',u'📓',u'📔',u'📒',u'📕',u'📗',u'📘',u'📙',u'📚',u'📇',u'🔗'
              ,u'📎',u'📌',u'📐',u'📍',u'📏',u'🚩',u'📁',u'📂',u'📝',u'🔏',u'🔐',u'🔒',u'🔓',u'📣',u'📢'
              ,u'🔈',u'🔉',u'🔊',u'🔇',u'💤',u'🔔',u'🔕',u'💭',u'💬',u'🚸',u'🔍',u'🔎',u'🎁',u'🚫',u'⛔'
              ,u'📛',u'🚷',u'🚯',u'🚳',u'🚱',u'📵',u'🔞',u'🉑',u'🉐',u'💮',u'㊗️',u'➕',u'➖',u'〰️',u'➗'
              ,u'🔃',u'💱',u'💲',u'➰',u'➿',u'〽️',u'❗',u'❓',u'❕',u'❔',u'❌',u'⭕',u'💯',u'🔚',u'🔙'
              ,u'🔛',u'🔝',u'🔜',u'🌀',u'Ⓜ️',u'⛎',u'🔯',u'🔰',u'🔱',u'⚠️',u'♻️',u'💢',u'💠',u'⚪️',u'⚫️'
              ,u'🔘',u'🔴',u'🔵',u'🔺',u'🔻',u'🔸',u'🔹',u'🔶',u'🔷',u'⬛',u'⬜',u'◾',u'◽️',u'🔲',u'🔳'
              ,u'🕐',u'🕜',u'🕑',u'🕝',u'🕒',u'🕞',u'🕓',u'🕟',u'🕔',u'🕠',u'🕕',u'🕡',u'🕖',u'🕢',u'🕗'
              ,u'🕣',u'🕘',u'🕤',u'🕙',u'🕥',u'🕦',u'🕛',u'🕧',u'🈴',u'🈵',u'🈲',u'🈶',u'🈚',u'🈸',u'🈺'
              ,u'🈹',u'🈳',u'🈁',u'🈯',u'💹',u'❎',u'✅',u'📳',u'📴',u'🆚',u'🅰️',u'🅱️',u'🆎',u'🆑',u'🅾️'
              ,u'🆘',u'🆔',u'🅿️',u'🚾',u'🆒',u'🆓',u'🆕',u'🆖',u'🆗',u'🆙',u'🏧',u'♈️',u'♉️',u'♊️',u'♋️'
              ,u'♌️',u'♍️',u'♎️',u'♏️',u'♐️',u'♑️',u'♒️',u'♓️',u'🚻',u'🚹',u'🚺',u'🚼',u'♿️',u'🚰',u'🚭'
              ,u'🚮',u'🔢',u'🔤',u'🔡',u'🔠',u'📶']

emoji_set = set(emoji_list)
# 絵文字検出正規表現
emoji_g = u"("+u"|".join(emoji_list)+u")"
emoji_p = re.compile(emoji_g)

# URL の正規表現
url_p = re.compile(u"https?:[A-z/.0-9]*")
# アカウントの正規表現
account_p = re.compile(u"@[A-z_0-9]*")
# RTの正規表現
rt_p = re.compile(u"RT.*")
# ハッシュタグの正規表現
tag_p = re.compile(ur'[#＃]([\w一-龠ぁ-んァ-ヴーａ-ｚ]+)')

count = 0

with gzip.open(sys.argv[1]) as f:
    for line in f:
        js = json.loads(line)
        tweet = js["text"]
        sentence = account_p.sub("",rt_p.sub("",tag_p.sub("",url_p.sub("",tweet))))
        temp = emoji_p.split(sentence)
        place = 0
        input_instance = []
        output_instance = []
        d = {}
        if len(temp) > 1:
            try:
                for each in temp:
                    if each in emoji_set:
                        output_instance.append((each,place))
                        if not output_instance or output_instance[-1][1] != place:
                            place += 1
                    else:
                        word_list = parse(each.encode("utf-8")).split()
                        for w in word_list:
                            input_instance.append(unicode(w))
                        place += len(word_list)
                d = {"input_instance":input_instance,"output_instance":output_instance}
#                print d
                count += 1
                js = json.dumps(d)
                print js
#                print u"元の文    ："+u" ".join(temp)
#                print u"絵文字以外："+u" ".join(input_instance)
#                print u"絵文字　　：",
#                for e,p in output_instance:
#                    print e,p,
#                print
#                print u"="*100
            except:
                pass
            if count == 1000000:
                break
