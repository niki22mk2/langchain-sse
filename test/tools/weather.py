import requests
import json

city_ids = {
    "横浜": "140010",
    "東京": "130010",
    "大阪": "270000",
    "名古屋": "230010",
    "札幌": "010010",
    "福岡": "400010",
    "仙台": "040010",
    "広島": "340010",
    "京都": "260010",
    "新潟": "150010",
    "神戸": "280010",
    "熊本": "430010",
}

forecast_days = {
    "今日": 0,
    "明日": 1,
    "明後日": 2,
}

def _extract_city_and_day(user_request):

    period = None
    city_name = None

    for day in forecast_days.keys():
        if day in user_request:
            period = day
            break

    for city in city_ids.keys():
        if city in user_request:
            city_name = city
            break

    return period, city_name


def get_weather(query):
    if "天気" not in query:
        return ""

    forecast_day, city_name = _extract_city_and_day(query)
    if city_name is None:
        city_name = "横浜"
    city_id = city_ids.get(city_name)

    url = f"https://weather.tsukumijima.net/api/forecast/city/{city_id}"
    response = requests.get(url)
    weather_data = json.loads(response.text)

    forecast_index = forecast_days.get(forecast_day, 0)

    target_forecast = weather_data["forecasts"][forecast_index]
    date = target_forecast["date"]
    weather = target_forecast["telop"]
    min_temp = target_forecast["temperature"]["min"]["celsius"]
    min_temp_str = f'{min_temp}°C' if min_temp else "--"
    max_temp = target_forecast["temperature"]["max"]["celsius"]
    max_temp_str = f'{max_temp}°C' if max_temp else "--"
    chance_of_rain = target_forecast["chanceOfRain"]

    def calculate_average_rain_chance(a, b):
        a = a.strip("%")
        b = b.strip("%")

        if a == "--" and b == "--":
            return "--"
        elif a == "--":
            return b
        elif b == "--":
            return a
        else:
            return (int(a) + int(b)) // 2

    morning_rain_chance = calculate_average_rain_chance(
        chance_of_rain["T00_06"], chance_of_rain["T06_12"]
    )
    afternoon_rain_chance = calculate_average_rain_chance(
        chance_of_rain["T12_18"], chance_of_rain["T18_24"]
    )

    formatted_weather = (
        f"{date}の{city_name}の天気は{weather}です。\n"
        f"最低気温：{min_temp_str}\n"
        f"最高気温：{max_temp_str}\n"
        f"降水確率：\n"
        f"  午前：{morning_rain_chance}%\n"
        f"  午後：{afternoon_rain_chance}%"
    )

    return formatted_weather

print(get_weather("横浜の明後日の天気"))