//! Daytime stress: our own take on Oura's "Daytime Stress".
//!
//! Oura's version runs a model over HR, HRV, motion and temperature that we don't
//! have. This keeps the same idea with transparent rules:
//!
//! 1. Waking hours only (beats inside a sleep window are skipped).
//! 2. 15-minute local-time slots; per slot, heart rate (median of good-quality
//!    inter-beat intervals) and RMSSD from successive beats within each burst,
//!    after dropping successive differences over 20 % (Malik artefact filter).
//! 3. Slots with real movement are `active` — exercise raises HR without stress.
//! 4. Quiet slots are scored against the wearer's own daytime baseline (median and
//!    MAD of the last 14 days): `index = z(HR) − z(ln RMSSD)`. High HR with low HRV
//!    reads as sympathetic load; the reverse as recovery.
//!
//! States by index: `stressed` > 1.5, `engaged` > 0.5, `restored` < −1.0, else
//! `relaxed`. Days roll up into minutes per state and a one-word summary.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use oura_store::storage::Store;
use serde_json::{json, Value};

use crate::ring_time::RingClock;

const SLOT_S: i64 = 15 * 60;
const MIN_BEATS: usize = 20;
/// Seconds of movement inside a slot that make it "active" rather than stress-scorable.
/// motion_event reports seconds moving per ~30 s window: sustained walking sums to
/// ~800 s per slot, fidgeting while seated or in bed to well under 100.
const ACTIVE_MOTION_S: f64 = 240.0;
const BASELINE_DAYS: i64 = 14;
/// Quiet slots needed before the baseline is trusted.
const MIN_BASELINE_SLOTS: usize = 12;

#[derive(Default)]
struct Slot {
    ibis: Vec<f64>,
    diffs: Vec<f64>,
    motion_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotStat {
    pub start: i64,
    pub hr: f64,
    pub rmssd: f64,
    pub active: bool,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Median and scaled MAD (≈ SD for normal data), floored so a flat baseline can't explode z.
fn robust(values: &[f64], floor: f64) -> (f64, f64) {
    let mut v = values.to_vec();
    let m = median(&mut v);
    let mut dev: Vec<f64> = values.iter().map(|x| (x - m).abs()).collect();
    (m, (1.4826 * median(&mut dev)).max(floor))
}

pub fn state(index: f64) -> &'static str {
    if index > 1.5 {
        "stressed"
    } else if index > 0.5 {
        "engaged"
    } else if index < -1.0 {
        "restored"
    } else {
        "relaxed"
    }
}

/// Classify slots against a baseline built from the quiet slots before each one.
pub fn classify(slots: &[SlotStat]) -> Vec<(SlotStat, &'static str, Option<f64>)> {
    slots
        .iter()
        .map(|s| {
            if s.active {
                return (*s, "active", None);
            }
            let window: Vec<&SlotStat> = slots
                .iter()
                .filter(|b| !b.active && b.start < s.start && b.start >= s.start - BASELINE_DAYS * 86_400)
                .collect();
            if window.len() < MIN_BASELINE_SLOTS {
                return (*s, "learning", None);
            }
            let (hr_m, hr_s) = robust(&window.iter().map(|b| b.hr).collect::<Vec<_>>(), 2.0);
            let (lr_m, lr_s) = robust(&window.iter().map(|b| b.rmssd.ln()).collect::<Vec<_>>(), 0.1);
            let index = (s.hr - hr_m) / hr_s - (s.rmssd.ln() - lr_m) / lr_s;
            (*s, state(index), Some(index))
        })
        .collect()
}

fn in_sleep(at: f64, sleeps: &[(f64, f64)]) -> bool {
    sleeps.iter().any(|(a, b)| at >= *a && at <= *b)
}

/// Daytime stress per local day. `nights` is the summary's nights array (for sleep windows).
pub fn daytime_stress(db: &Path, tz: f64, nights: &Value) -> Result<Value> {
    let sleeps: Vec<(f64, f64)> = nights
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| Some((n["start_unix"].as_f64()?, n["end_unix"].as_f64()?)))
        .collect();
    let store = Store::open_read_only(db).context("opening DB")?;
    let events = store.decoded_events().context("reading events")?;
    let clock = RingClock::from_events(&events);
    let off = crate::offset_seconds(tz);
    let slot_of = |at: f64| ((at as i64 + off).div_euclid(SLOT_S)) * SLOT_S - off;

    let mut slots: BTreeMap<i64, Slot> = BTreeMap::new();
    for (ds, tag, jstr, cu) in &events {
        let name = oura_protocol::events::event_name(*tag);
        if !matches!(name, "green_ibi_quality_event" | "motion_event") {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(jstr) else { continue };
        let at = clock.unix_s(*ds, *cu);
        if !at.is_finite() || in_sleep(at, &sleeps) {
            continue;
        }
        let slot = slots.entry(slot_of(at)).or_default();
        if name == "motion_event" {
            slot.motion_s += v["motion_seconds"].as_f64().unwrap_or(0.0);
            continue;
        }
        let ibis: Vec<f64> = v["ibi_ms"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
        let quality: Vec<i64> = v["quality"].as_array().into_iter().flatten().filter_map(Value::as_i64).collect();
        let good: Vec<f64> = ibis
            .iter()
            .enumerate()
            .filter(|(i, ibi)| quality.get(*i).copied().unwrap_or(1) >= 1 && (300.0..=2000.0).contains(*ibi))
            .map(|(_, ibi)| *ibi)
            .collect();
        for w in good.windows(2) {
            if (w[1] - w[0]).abs() <= 0.2 * w[0] {
                slot.diffs.push(w[1] - w[0]);
            }
        }
        slot.ibis.extend(good);
    }

    let stats: Vec<SlotStat> = slots
        .into_iter()
        .filter(|(_, s)| s.ibis.len() >= MIN_BEATS && s.diffs.len() >= MIN_BEATS / 2)
        .map(|(start, mut s)| {
            let hr = 60_000.0 / median(&mut s.ibis);
            let rmssd = (s.diffs.iter().map(|d| d * d).sum::<f64>() / s.diffs.len() as f64).sqrt().max(1.0);
            SlotStat { start, hr, rmssd, active: s.motion_s >= ACTIVE_MOTION_S || hr > 110.0 }
        })
        .collect();

    let mut days: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (s, st, idx) in classify(&stats) {
        let local = s.start + off;
        let (y, m, d) = crate::civil(local.div_euclid(86_400));
        let hm = local.rem_euclid(86_400);
        days.entry(format!("{y:04}-{m:02}-{d:02}")).or_default().push(json!({
            "unix": s.start,
            "hm": format!("{:02}:{:02}", hm / 3600, hm % 3600 / 60),
            "state": st,
            "hr": s.hr.round(),
            "rmssd": s.rmssd.round(),
            "index": idx.map(|i| (i * 100.0).round() / 100.0),
        }));
    }

    let out: Vec<Value> = days
        .into_iter()
        .map(|(date, slots)| {
            let mins = |st: &str| slots.iter().filter(|s| s["state"] == st).count() as i64 * (SLOT_S / 60);
            let (stressed, restored) = (mins("stressed"), mins("restored"));
            let scored = stressed + restored + mins("engaged") + mins("relaxed");
            let summary = if scored == 0 || mins("learning") > scored {
                "learning"
            } else if stressed >= 90 && stressed > 2 * restored {
                "stressful"
            } else if restored >= 60 && restored > stressed {
                "restored"
            } else {
                "normal"
            };
            json!({
                "date": date,
                "summary": summary,
                "stressed_min": stressed, "engaged_min": mins("engaged"), "relaxed_min": mins("relaxed"),
                "restored_min": restored, "active_min": mins("active"),
                "slots": slots,
            })
        })
        .collect();
    Ok(json!({ "slot_minutes": SLOT_S / 60, "days": out }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(start: i64, hr: f64, rmssd: f64) -> SlotStat {
        SlotStat { start, hr, rmssd, active: false }
    }

    #[test]
    fn needs_a_baseline_first() {
        let slots: Vec<SlotStat> = (0..5).map(|i| quiet(i * SLOT_S, 60.0, 40.0)).collect();
        assert!(classify(&slots).iter().all(|(_, st, _)| *st == "learning"));
    }

    #[test]
    fn high_hr_low_hrv_is_stress_and_the_reverse_restores() {
        let mut slots: Vec<SlotStat> =
            (0..30).map(|i| quiet(i * SLOT_S, 62.0 + (i % 3) as f64, 40.0 + (i % 4) as f64)).collect();
        slots.push(quiet(31 * SLOT_S, 80.0, 18.0));
        slots.push(quiet(32 * SLOT_S, 54.0, 75.0));
        slots.push(SlotStat { start: 33 * SLOT_S, hr: 120.0, rmssd: 10.0, active: true });
        let out = classify(&slots);
        let st = |i: usize| out[out.len() - 3 + i].1;
        assert_eq!(st(0), "stressed");
        assert_eq!(st(1), "restored");
        assert_eq!(st(2), "active");
    }

    #[test]
    fn robust_floor_prevents_division_blowups() {
        let (_, s) = robust(&[60.0; 20], 2.0);
        assert_eq!(s, 2.0);
    }
}
