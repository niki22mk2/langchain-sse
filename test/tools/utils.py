from datetime import datetime, timedelta, timezone

def get_date(raw=False):
    JST = timezone(timedelta(hours=+9), "JST")

    weekday_ja = {
        'Mon': '月曜日',
        'Tue': '火曜日',
        'Wed': '水曜日',
        'Thu': '木曜日',
        'Fri': '金曜日',
        'Sat': '土曜日',
        'Sun': '日曜日'
    }

    now = datetime.now(JST)
    date_str = now.strftime('%Y/%m/%d')
    weekday_en = now.strftime('%a')
    weekday_ja_str = weekday_ja[weekday_en]
    time_str = now.strftime('%H:%M')

    date_str = f'{date_str} {weekday_ja_str} {time_str}'

    if raw:
        return now
    else:
        return date_str