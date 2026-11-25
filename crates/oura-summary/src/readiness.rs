//! A daily Readiness score, built on top of the finished summary.
//!
//! Oura's exact contributor curves are calibrated from a private trends export (see
//! `tools/score_readiness.py`), so this is our own score with the **same structure**:
//! the eight contributors and their weights recovered in `docs/algorithms/score-weights.md`,
//! each mapped to 0–100 by a transparent piecewise-linear curve. Physiology is judged
//! against the wearer's own trailing baseline, not population norms.
//!
//! Each contributor needs its own inputs; one without data drops out and the rest
//! renormalise (same rule as `sleep_score`). Until [`MATURE_NIGHTS`] main sleeps exist
//! the baselines are thin, so the result is flagged `provisional`.

use crate::sleep_score::curve;
use serde_json::{json, Value};

/// Nights of history before baselines are considered settled (Oura uses ~2 weeks).
pub const MATURE_NIGHTS: usize = 14;
/// Trailing window for baselines.
const WINDOW: usize = 14;
/// Shortest sleep that counts as a main sleep (naps don't move readiness baselines).
const MIN_MAIN_SLEEP_H: f64 = 3.0;
/// Daily active-calorie goal used for the activity contributors until goals exist.
const ACTIVE_KCAL_TARGET: f64 = 400.0;
const SLEEP_NEED_H: f64 = 8.0;

/// The weights recovered from Oura's combiner (sum 99 → renormalised anyway).
const WEIGHTS: [(&str, f64); 8] = [
    ("resting_hr", 17.0),
    ("previous_night", 15.0),
    ("hrv_balance", 15.0),
    ("temperature", 13.0),
    ("sleep_balance", 12.0),
    ("previous_day_activity", 10.0),
    ("recovery_index", 10.0),
    ("activity_balance", 7.0),
];

#[derive(Clone, Default)]
struct Night {
    day: String,
    rhr: Option<f64>,
    hrv: Option<f64>,
    temp: Option<f64>,
    asleep_h: f64,
    sleep_score: Option<f64>,
    /// Hours between the night's lowest heart rate and waking.
    lowest_hr_before_wake_h: Option<f64>,
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite())
}

fn mean(xs: &[f64]) -> Option<f64> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

fn sd(xs: &[f64]) -> Option<f64> {
    let m = mean(xs)?;
    (xs.len() >= 3).then(|| (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt())
}

fn median(xs: &[f64]) -> Option<f64> {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    (!v.is_empty()).then(|| v[v.len() / 2])
}

/// Where in the night the heart rate bottomed out, as hours before waking. Uses a
/// 5-sample moving average so one low outlier beat can't decide it.
fn lowest_hr_before_wake(n: &Value) -> Option<f64> {
    let hr: Vec<f64> = n["series"]["hr"].as_array()?.iter().map(|v| num(v).unwrap_or(0.0)).collect();
    let dur_h = match (num(&n["start_unix"]), num(&n["end_unix"])) {
        (Some(a), Some(b)) if b > a => (b - a) / 3600.0,
        _ => num(&n["in_bed_h"])?,
    };
    if hr.iter().filter(|v| **v > 0.0).count() < 10 {
        return None;
    }
    let k = 5;
    let mut best: Option<(usize, f64)> = None;
    for i in 0..hr.len().saturating_sub(k - 1) {
        let w: Vec<f64> = hr[i..i + k].iter().copied().filter(|v| *v > 0.0).collect();
        if w.len() < 3 {
            continue;
        }
        let m = w.iter().sum::<f64>() / w.len() as f64;
        if best.is_none_or(|(_, b)| m < b) {
            best = Some((i + k / 2, m));
        }
    }
    let (idx, _) = best?;
    Some(dur_h * (1.0 - idx as f64 / (hr.len() - 1).max(1) as f64))
}

fn main_nights(summary: &Value) -> Vec<Night> {
    let mut out: Vec<(i64, Night)> = summary["nights"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| {
            let in_bed = num(&n["in_bed_h"])?;
            if in_bed < MIN_MAIN_SLEEP_H {
                return None;
            }
            let day = n["wake_ymd"].as_str().or(n["date"].as_str())?.to_owned();
            let asleep_min = num(&n["metrics"]["asleep_min"]).or(num(&n["metrics"]["total_sleep_min"]));
            Some((
                n["start_unix"].as_i64().unwrap_or(0),
                Night {
                    day,
                    rhr: num(&n["rhr"]),
                    hrv: num(&n["hrv_ms"]),
                    temp: num(&n["skin_temp"]),
                    asleep_h: asleep_min.map(|m| m / 60.0).unwrap_or(in_bed),
                    sleep_score: num(&n["sleep_score"]["score"]),
                    lowest_hr_before_wake_h: lowest_hr_before_wake(n),
                },
            ))
        })
        .collect();
    out.sort_by_key(|(t, _)| *t);
    // one main sleep per wake day: keep the longest
    let mut by_day: Vec<Night> = Vec::new();
    for (_, n) in out {
        match by_day.last_mut() {
            Some(last) if last.day == n.day => {
                if n.asleep_h > last.asleep_h {
                    *last = n;
                }
            }
            _ => by_day.push(n),
        }
    }
    by_day
}

fn active_kcal(summary: &Value, day: &str) -> Option<f64> {
    num(&summary["activity_daily"][day]["active_kcal"])
}

fn prev_day(day: &str) -> Option<String> {
    let d = chrono_like_parse(day)?;
    let (y, m, dd) = crate::civil(d - 1);
    Some(format!("{y:04}-{m:02}-{dd:02}"))
}

fn chrono_like_parse(day: &str) -> Option<i64> {
    let mut p = day.split('-');
    let y = p.next()?.parse().ok()?;
    let m = p.next()?.parse().ok()?;
    let d = p.next()?.parse().ok()?;
    Some(crate::days_from_civil(y, m, d))
}

/// Score one wake day given the nights up to and including it.
fn score_day(summary: &Value, history: &[Night]) -> Value {
    let today = history.last().expect("non-empty");
    let prior: &[Night] = &history[history.len().saturating_sub(WINDOW + 1)..history.len() - 1];
    let prior_of = |f: fn(&Night) -> Option<f64>| prior.iter().filter_map(f).collect::<Vec<_>>();
    let mut comps: Vec<(&str, f64, Value)> = Vec::new();

    // Resting HR: last night's vs your baseline; every bpm above costs points.
    let rhr_base = prior_of(|n| n.rhr);
    if let (Some(rhr), Some(base)) = (today.rhr, median(&rhr_base).filter(|_| rhr_base.len() >= 3)) {
        let d = rhr - base;
        comps.push(("resting_hr", curve(d, &[(-3.0, 100.0), (0.0, 92.0), (3.0, 70.0), (6.0, 40.0), (10.0, 10.0)]),
            json!({"value": rhr.round(), "baseline": (base * 10.0).round() / 10.0})));
    }

    // Previous night: that night's sleep score, or its duration if unscored.
    let prev_night = today.sleep_score.unwrap_or_else(|| curve(today.asleep_h, &[(4.0, 20.0), (6.0, 60.0), (7.0, 85.0), (8.0, 100.0)]));
    comps.push(("previous_night", prev_night, json!({"asleep_h": (today.asleep_h * 10.0).round() / 10.0})));

    // HRV balance: recent 3-night HRV vs baseline, in baseline standard deviations.
    let hrv_base = prior_of(|n| n.hrv);
    let recent: Vec<f64> = history.iter().rev().take(3).filter_map(|n| n.hrv).collect();
    if let (Some(r), Some(m), Some(s)) = (mean(&recent), mean(&hrv_base), sd(&hrv_base)) {
        let z = (r - m) / s.max(1.0);
        comps.push(("hrv_balance", curve(z, &[(-2.0, 20.0), (-1.0, 60.0), (-0.3, 85.0), (0.3, 100.0)]),
            json!({"recent_ms": r.round(), "baseline_ms": m.round()})));
    }

    // Temperature: deviation from your baseline either way is a strain signal.
    let temp_base = prior_of(|n| n.temp);
    if let (Some(t), Some(base)) = (today.temp, median(&temp_base).filter(|_| temp_base.len() >= 3)) {
        let dev = t - base;
        comps.push(("temperature", curve(dev.abs(), &[(0.1, 100.0), (0.3, 85.0), (0.6, 55.0), (1.0, 20.0)]),
            json!({"deviation_c": (dev * 100.0).round() / 100.0})));
    }

    // Sleep balance: two weeks of sleep against need.
    let recent_sleep: Vec<f64> = history.iter().rev().take(WINDOW).map(|n| n.asleep_h).collect();
    if recent_sleep.len() >= 3 {
        let avg = mean(&recent_sleep).unwrap_or(0.0);
        comps.push(("sleep_balance", curve(avg / SLEEP_NEED_H, &[(0.6, 20.0), (0.8, 60.0), (0.9, 85.0), (1.0, 100.0)]),
            json!({"avg_asleep_h": (avg * 10.0).round() / 10.0})));
    }

    // Previous day activity: too little and too much both cost (Oura's shape).
    if let Some(kcal) = prev_day(&today.day).and_then(|d| active_kcal(summary, &d)) {
        let r = kcal / ACTIVE_KCAL_TARGET;
        comps.push(("previous_day_activity", curve(r, &[(0.0, 50.0), (0.5, 85.0), (0.8, 100.0), (1.5, 100.0), (2.5, 60.0), (4.0, 30.0)]),
            json!({"active_kcal": kcal.round()})));
    }

    // Recovery index: how long before waking your heart rate bottomed out.
    if let Some(h) = today.lowest_hr_before_wake_h {
        comps.push(("recovery_index", curve(h, &[(0.5, 20.0), (2.0, 50.0), (4.0, 80.0), (6.0, 100.0)]),
            json!({"lowest_hr_hours_before_wake": (h * 10.0).round() / 10.0})));
    }

    // Activity balance: two weeks of activity against target.
    let days: Vec<f64> = history.iter().rev().take(WINDOW).filter_map(|n| active_kcal(summary, &n.day)).collect();
    if days.len() >= 3 {
        let r = mean(&days).unwrap_or(0.0) / ACTIVE_KCAL_TARGET;
        comps.push(("activity_balance", curve(r, &[(0.2, 30.0), (0.5, 65.0), (0.8, 90.0), (1.0, 100.0), (2.0, 100.0), (3.0, 75.0)]),
            json!({"avg_active_kcal": (r * ACTIVE_KCAL_TARGET).round()})));
    }

    let weight = |k: &str| WEIGHTS.iter().find(|(w, _)| *w == k).map(|(_, v)| *v).unwrap_or(0.0);
    let total_w: f64 = comps.iter().map(|(k, _, _)| weight(k)).sum();
    let score = comps.iter().map(|(k, s, _)| s * weight(k)).sum::<f64>() / total_w.max(1e-9);
    json!({
        "date": today.day,
        "score": score.round(),
        "provisional": history.len() < MATURE_NIGHTS,
        "nights_of_history": history.len(),
        "contributors": comps.iter().map(|(k, s, d)| json!({
            "key": k, "score": s.round(), "weight": weight(k), "detail": d,
        })).collect::<Vec<_>>(),
    })
}

/// Readiness for every wake day with a main sleep, oldest first.
pub fn score_days(summary: &Value) -> Value {
    let nights = main_nights(summary);
    json!((1..=nights.len()).map(|i| score_day(summary, &nights[..i])).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn night(day: &str, start: i64, rhr: f64, hrv: f64, temp: f64) -> Value {
        json!({"wake_ymd": day, "start_unix": start, "end_unix": start + 8 * 3600, "in_bed_h": 8.0,
               "rhr": rhr, "hrv_ms": hrv, "skin_temp": temp, "sleep_score": {"score": 80.0}})
    }

    fn summary(nights: Vec<Value>) -> Value {
        json!({"nights": nights, "activity_daily": {}})
    }

    #[test]
    fn steady_nights_score_well_and_a_bad_night_drops() {
        let mut ns: Vec<Value> = (0..14)
            .map(|i| night(&format!("2026-11-{:02}", i + 1), i * 86_400, 52.0 + (i % 2) as f64, 70.0 + (i % 3) as f64, 35.0))
            .collect();
        let good = score_days(&summary(ns.clone()));
        let g = good.as_array().unwrap().last().unwrap().clone();
        assert!(g["score"].as_f64().unwrap() >= 80.0, "{g}");
        assert_eq!(g["provisional"], false);

        ns.push(night("2026-11-15", 14 * 86_400, 61.0, 45.0, 35.8));
        let bad = score_days(&summary(ns));
        let b = bad.as_array().unwrap().last().unwrap().clone();
        assert!(b["score"].as_f64().unwrap() < g["score"].as_f64().unwrap() - 15.0, "{b}");
    }

    #[test]
    fn first_night_is_provisional_and_skips_baseline_contributors() {
        let out = score_days(&summary(vec![night("2026-11-01", 0, 52.0, 70.0, 35.0)]));
        let d = &out[0];
        assert_eq!(d["provisional"], true);
        let keys: Vec<&str> = d["contributors"].as_array().unwrap().iter().map(|c| c["key"].as_str().unwrap()).collect();
        assert!(!keys.contains(&"resting_hr") && !keys.contains(&"hrv_balance"), "{keys:?}");
        assert!(keys.contains(&"previous_night"));
    }

    #[test]
    fn naps_are_ignored() {
        let mut nap = night("2026-11-01", 50_000, 70.0, 20.0, 36.0);
        nap["in_bed_h"] = json!(1.0);
        let out = score_days(&summary(vec![nap]));
        assert_eq!(out.as_array().unwrap().len(), 0);
    }
}
