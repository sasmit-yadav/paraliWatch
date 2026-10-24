"""Plain-language text for SHAP reasons, so officials don't have to read feature names."""

from __future__ import annotations

TEMPLATES = {
    "fires_1d": "{v:.0f} fire(s) detected in this cell today",
    "fires_3d": "{v:.0f} fire(s) in this cell over the last 3 days",
    "fires_7d": "{v:.0f} fire(s) in this cell over the last 7 days",
    "fires_14d": "{v:.0f} fire(s) in this cell over the last 14 days",
    "fires_season": "{v:.0f} fire(s) in this cell so far this season",
    "frp_3d": "Fire intensity in this cell over 3 days: {v:.0f} MW",
    "days_since_fire": "Days since the last fire here: {v:.0f}",
    "ring1_1d": "{v:.0f} fire(s) in the 6 surrounding cells today",
    "ring1_3d": "{v:.0f} fire(s) in the 6 surrounding cells over 3 days",
    "ring1_7d": "{v:.0f} fire(s) in the 6 surrounding cells over 7 days",
    "ring2_3d": "{v:.0f} fire(s) in the next ring of cells (~10 km) over 3 days",
    "district_3d": "{v:.0f} fire(s) in this district over 3 days",
    "state_3d": "{v:.0f} fire(s) across the state over 3 days",
    "state_season_vs_last_year": "State's fires so far vs last year at this date: {v:.2f}x",
    "last_year_next3": "{v:.0f} fire(s) here on the same days last year",
    "last_year_week": "{v:.0f} fire(s) here in the same week last year",
    "last_year_season": "{v:.0f} fire(s) here during all of last season",
    "history_rate": "This cell averages {v:.0f} fire(s) per season in past years",
    "season_day": "Timing in the season (day {v:.0f} after 15 Sep)",
    "tmax": "Max temperature {v:.0f} C",
    "rh": "Humidity {v:.0f}%",
    "rain": "Rain today {v:.1f} mm",
    "rain_3d": "Rain over 3 days {v:.1f} mm",
    "wind": "Max wind {v:.0f} km/h",
    "wind_dir_sin": "Wind direction",
    "wind_dir_cos": "Wind direction",
}


def describe(feature: str, value: float | None) -> str:
    """Sentence for one reason. Without a value, fall back to the feature's plain name."""
    template = TEMPLATES.get(feature)
    if template is None:
        return feature.replace("_", " ")
    if "{v" not in template:
        return template
    if value is None:
        return feature.replace("_", " ")
    if feature == "days_since_fire" and value == 0:
        return "A fire was detected in this cell today"
    return template.format(v=float(value))
