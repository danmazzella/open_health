//! Rule-based activity sessions, used when Oura's `automatic_activity_detection`
//! model isn't available. Same JSON shape as `tools/run_activity_model.py` sessions
//! so every client renders either source.
//!
//! A minute is "moving" at ≥ 3 MET (brisk walking, ACSM's moderate threshold). Runs of
//! moving minutes that bridge pauses of ≤ 3 min and last ≥ 10 min become sessions,
//! labelled by mean intensity: < 5 MET walking, 5–7 workout, ≥ 7 running. Without the
//! gait channel the label is a guess from intensity alone, so `label_confidence` stays null.

use std::collections::BTreeMap;

use serde_json::{json, Value};

const MOVING_MET_ABOVE_REST: f64 = 2.0; // 3 MET total
const MAX_GAP_MIN: i64 = 3;
const MIN_SESSION_MIN: i64 = 10;

/// `met_above_rest` is keyed by local minute index (local unix / 60).
pub fn detect(met_above_rest: &BTreeMap<i64, f64>) -> Vec<Value> {
    let moving: Vec<i64> = met_above_rest
        .iter()
        .filter(|(_, m)| **m >= MOVING_MET_ABOVE_REST)
        .map(|(k, _)| *k)
        .collect();
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for m in moving {
        match runs.last_mut() {
            Some((_, end)) if m - *end <= MAX_GAP_MIN + 1 => *end = m,
            _ => runs.push((m, m)),
        }
    }
    runs.into_iter()
        .filter(|(a, b)| b - a + 1 >= MIN_SESSION_MIN)
        .map(|(a, b)| {
            let mets: Vec<f64> = (a..=b).filter_map(|m| met_above_rest.get(&m)).map(|v| v + 1.0).collect();
            let mean = mets.iter().sum::<f64>() / mets.len().max(1) as f64;
            let label = if mean >= 7.0 {
                "running"
            } else if mean >= 5.0 {
                "workout"
            } else {
                "walking"
            };
            let dur = b - a + 1;
            let (y, mo, d) = crate::civil(a.div_euclid(1440));
            let hm = |m: i64| format!("{:02}:{:02}", m.rem_euclid(1440) / 60, m.rem_euclid(60));
            json!({
                "start": format!("{y:04}-{mo:02}-{d:02} {}", hm(a)),
                "end": hm(b + 1),
                "duration_min": dur,
                // sustained and intense reads as a deliberate workout; a short stroll less so
                "is_workout": ((mean - 3.0) / 4.0 + dur as f64 / 60.0).clamp(0.2, 1.0),
                "label": label,
                "label_confidence": Value::Null,
                "mean_met": (mean * 10.0).round() / 10.0,
                "source": "rules",
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minutes(start: i64, n: i64, met_above: f64, map: &mut BTreeMap<i64, f64>) {
        for m in start..start + n {
            map.insert(m, met_above);
        }
    }

    #[test]
    fn finds_a_run_and_a_walk_and_ignores_blips() {
        let day = crate::days_from_civil(2026, 11, 25) * 1440;
        let mut met = BTreeMap::new();
        minutes(day + 7 * 60, 30, 8.0, &mut met); // 07:00 run, 9 MET
        minutes(day + 12 * 60, 12, 2.5, &mut met); // 12:00 walk
        minutes(day + 12 * 60 + 14, 10, 2.5, &mut met); // continues after a 2-min pause
        minutes(day + 15 * 60, 4, 6.0, &mut met); // too short
        let s = detect(&met);
        assert_eq!(s.len(), 2, "{s:?}");
        assert_eq!(s[0]["label"], "running");
        assert_eq!(s[0]["start"], "2026-11-25 07:00");
        assert_eq!(s[0]["duration_min"], 30);
        assert_eq!(s[1]["label"], "walking");
        assert_eq!(s[1]["duration_min"], 24);
    }
}
