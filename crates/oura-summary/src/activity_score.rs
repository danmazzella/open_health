//! A daily Activity score with Oura's structure: the five contributors and weights
//! recovered in `docs/algorithms/score-weights.md`, each through a transparent
//! piecewise curve (Oura's own curves are calibrated from a private export).
//!
//! Inputs are the finished summary's `activity_profile` (96 × 15-min MET above rest)
//! and `activity_daily` (active kcal). Waking slots are those outside the night's
//! sleep window, falling back to 07:00–23:00 when no night is known. A slot with no
//! movement data counts as inactive, so a day with the ring off scores low on the
//! inactivity contributors — the same thing Oura does.

use crate::sleep_score::curve;
use serde_json::{json, Value};

const WEIGHTS: [(&str, f64); 5] = [
    ("move_every_hour", 33.0),
    ("meet_daily_targets", 24.0),
    ("stay_active", 17.0),
    ("training_volume", 15.0),
    ("training_frequency", 12.0),
];
const ACTIVE_KCAL_TARGET: f64 = 400.0;
/// MET above rest that counts as "not sitting still" (light activity, ~1.5 MET).
const MOVING: f64 = 0.5;
/// MET above rest for moderate-or-harder training (~3 MET).
const MODERATE: f64 = 2.0;

fn num(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite())
}

/// Which of the day's 96 slots are waking hours.
fn waking_slots(summary: &Value, day: &str) -> Vec<usize> {
    let tz_s = num(&summary["tz"]).unwrap_or(0.0) * 3600.0;
    let Some(day_idx) = parse_day(day) else { return (28..92).collect() };
    let day_start = day_idx as f64 * 86_400.0 - tz_s; // unix of local midnight
    let sleeps: Vec<(f64, f64)> = summary["nights"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| Some((num(&n["start_unix"])?, num(&n["end_unix"])?)))
        .filter(|(a, b)| *b > day_start && *a < day_start + 86_400.0)
        .collect();
    if sleeps.is_empty() {
        return (28..92).collect(); // 07:00–23:00
    }
    (0..96)
        .filter(|i| {
            let mid = day_start + (*i as f64 + 0.5) * 900.0;
            !sleeps.iter().any(|(a, b)| mid >= *a && mid <= *b)
        })
        .collect()
}

fn parse_day(day: &str) -> Option<i64> {
    let mut p = day.split('-');
    Some(crate::days_from_civil(p.next()?.parse().ok()?, p.next()?.parse().ok()?, p.next()?.parse().ok()?))
}

fn profile(summary: &Value, day: &str) -> Option<Vec<f64>> {
    let v: Vec<f64> = summary["activity_profile"][day].as_array()?.iter().map(|x| num(x).unwrap_or(0.0)).collect();
    (v.len() == 96).then_some(v)
}

fn moderate_min(summary: &Value, day: &str) -> f64 {
    profile(summary, day).map_or(0.0, |p| p.iter().filter(|m| **m >= MODERATE).count() as f64 * 15.0)
}

fn score_day(summary: &Value, day: &str, prior_days: &[String]) -> Option<Value> {
    let prof = profile(summary, day)?;
    let waking = waking_slots(summary, day);
    let mut comps: Vec<(&str, f64, Value)> = Vec::new();

    // Stay active: hours of waking inactivity.
    let inactive_h = waking.iter().filter(|i| prof[**i] < MOVING).count() as f64 / 4.0;
    comps.push(("stay_active", curve(inactive_h, &[(5.0, 100.0), (8.0, 80.0), (10.0, 55.0), (12.0, 25.0), (14.0, 5.0)]),
        json!({"inactive_h": inactive_h})));

    // Move every hour: waking hours with no movement at all in any of their 4 slots.
    let mut still_hours = 0;
    for h in 0..24 {
        let slots: Vec<usize> = (h * 4..h * 4 + 4).filter(|i| waking.contains(i)).collect();
        if slots.len() == 4 && slots.iter().all(|i| prof[*i] < MOVING) {
            still_hours += 1;
        }
    }
    comps.push(("move_every_hour", curve(still_hours as f64, &[(0.0, 100.0), (1.0, 90.0), (3.0, 65.0), (5.0, 35.0), (8.0, 5.0)]),
        json!({"inactive_hours": still_hours})));

    // Meet daily targets: active calories vs goal.
    let kcal = num(&summary["activity_daily"][day]["active_kcal"]);
    if let Some(k) = kcal {
        comps.push(("meet_daily_targets", curve(k / ACTIVE_KCAL_TARGET, &[(0.0, 0.0), (0.25, 25.0), (0.75, 80.0), (1.0, 100.0)]),
            json!({"active_kcal": k.round(), "target_kcal": ACTIVE_KCAL_TARGET})));
    }

    // Training volume and frequency over the last 7 days (this one included).
    let mut days7: Vec<String> = prior_days.iter().rev().take(6).cloned().collect();
    days7.push(day.to_owned());
    let vol: f64 = days7.iter().filter_map(|d| num(&summary["activity_daily"][d.as_str()]["active_kcal"])).sum();
    let ratio = vol / (ACTIVE_KCAL_TARGET * 7.0);
    comps.push(("training_volume", curve(ratio, &[(0.0, 10.0), (0.4, 50.0), (0.8, 90.0), (1.0, 100.0)]),
        json!({"week_active_kcal": vol.round()})));
    let train_days = days7.iter().filter(|d| moderate_min(summary, d) >= 30.0).count();
    comps.push(("training_frequency", curve(train_days as f64, &[(0.0, 20.0), (1.0, 45.0), (2.0, 65.0), (3.0, 85.0), (4.0, 100.0)]),
        json!({"training_days": train_days})));

    let weight = |k: &str| WEIGHTS.iter().find(|(w, _)| *w == k).map(|(_, v)| *v).unwrap_or(0.0);
    let total_w: f64 = comps.iter().map(|(k, _, _)| weight(k)).sum();
    let score = comps.iter().map(|(k, s, _)| s * weight(k)).sum::<f64>() / total_w;
    Some(json!({
        "date": day,
        "score": score.round(),
        "provisional": prior_days.len() < 6, // a full week before volume/frequency mean anything
        "contributors": comps.iter().map(|(k, s, d)| json!({"key": k, "score": s.round(), "weight": weight(k), "detail": d})).collect::<Vec<_>>(),
    }))
}

/// Activity score for each day with a movement profile, oldest first.
pub fn score_days(summary: &Value) -> Value {
    let mut days: Vec<String> = summary["activity_profile"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    days.sort();
    json!((0..days.len()).filter_map(|i| score_day(summary, &days[i], &days[..i])).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day_profile(active_slots: &[usize], level: f64) -> Vec<f64> {
        (0..96).map(|i| if active_slots.contains(&i) { level } else { 0.0 }).collect()
    }

    #[test]
    fn an_active_day_beats_a_sedentary_one() {
        // active: something every hour 7–23 plus a 45-min workout
        let mut active: Vec<usize> = (28..92).step_by(4).collect();
        active.extend(72..75);
        let s = json!({
            "tz": 0.0, "nights": [],
            "activity_profile": {
                "2026-11-24": day_profile(&active, 3.0),
                "2026-11-25": day_profile(&[], 0.0),
            },
            "activity_daily": {"2026-11-24": {"active_kcal": 450.0}, "2026-11-25": {"active_kcal": 40.0}},
        });
        let out = score_days(&s);
        let a = out[0]["score"].as_f64().unwrap();
        let b = out[1]["score"].as_f64().unwrap();
        assert!(a > b + 40.0 && b < 30.0, "active {a}, sedentary {b}: {out}");
        assert_eq!(out[1]["provisional"], true);
    }
}
