//! UTC formatting without an extra time crate.

use std::time::{SystemTime, UNIX_EPOCH};

/// Format `time` as `YYYY-MM-DDTHH:MM:SS.mmmZ`.
pub(crate) fn format_rfc3339(time: SystemTime) -> String {
    let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    format_rfc3339_parts(duration.as_secs(), duration.subsec_millis())
}

pub(crate) fn format_rfc3339_parts(secs: u64, millis: u32) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Howard Hinnant's `civil_from_days`, shifted so day 0 is the Unix epoch.
fn civil_from_unix(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = i64::try_from(secs).unwrap_or(i64::MAX);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400) as u32;
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year, month as u32, day as u32, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_epoch_and_a_known_second_are_rfc3339() {
        assert_eq!(format_rfc3339_parts(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            format_rfc3339_parts(1_000_000_000, 0),
            "2001-09-09T01:46:40.000Z"
        );
        assert_eq!(
            format_rfc3339_parts(1_700_000_000, 123),
            "2023-11-14T22:13:20.123Z"
        );
    }
}
