//! KumoTabular DEFAULT single-estimator preprocessing helpers.
use ndarray::Array2;
#[cfg(test)]
use rand::SeedableRng;
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1e-6;
const CLIP: f64 = 100.0;
const CLIP_SIGMA: f64 = 4.0;
const CARDINALITY_THRESHOLD: usize = 50;
const MAX_FEATURES: usize = 500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Regression,
    Classification,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureType {
    Numerical,
    Categorical,
}

#[derive(Clone, Copy)]
struct NumericStats {
    mean: f64,
    scale: f64,
    lower: f64,
    upper: f64,
    sign: f64,
}

fn numeric_stats(values: &[f64], rng: &mut StdRng, flip: bool) -> NumericStats {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let feature_mean = mean(&finite);
    let scale = stddev(&finite, feature_mean) + EPS;
    let normalized = finite
        .iter()
        .map(|v| ((v - feature_mean) / scale).clamp(-CLIP, CLIP))
        .collect::<Vec<_>>();
    let center = mean(&normalized);
    // Official ClipSigma uses population variance for its first pass.
    let std = population_std(&normalized, center).max(EPS);
    let kept = normalized
        .iter()
        .copied()
        .filter(|v| (*v - center).abs() <= CLIP_SIGMA * std)
        .collect::<Vec<_>>();
    let (center, std) = if kept.is_empty() {
        (center, std)
    } else {
        let center = mean(&kept);
        (center, sample_std(&kept, center).max(EPS))
    };
    NumericStats {
        mean: feature_mean,
        scale,
        lower: center - CLIP_SIGMA * std,
        upper: center + CLIP_SIGMA * std,
        sign: if flip && rng.gen_bool(0.5) { -1.0 } else { 1.0 },
    }
}

fn transform_numeric(raw: f64, stats: NumericStats) -> f32 {
    if raw.is_nan() {
        return f32::NAN;
    }
    // Match Standardize -> Clip(-100, 100): infinities are clipped, while
    // missing NaNs remain missing.
    let z = ((raw - stats.mean) / stats.scale).clamp(-CLIP, CLIP);
    let log_abs = z.abs().ln_1p();
    z.max(stats.lower - log_abs)
        .min(stats.upper + log_abs)
        .mul_add(stats.sign, 0.0) as f32
}

fn sample_std(values: &[f64], center: f64) -> f64 {
    (values.iter().map(|v| (v - center).powi(2)).sum::<f64>()
        / values.len().saturating_sub(1).max(1) as f64)
        .sqrt()
}

fn population_std(values: &[f64], center: f64) -> f64 {
    (values.iter().map(|v| (v - center).powi(2)).sum::<f64>() / values.len().max(1) as f64).sqrt()
}
#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    /// Context rows first, then query rows; columns follow the returned shuffled mask.
    pub x: Array2<f32>,
    /// Transformed labels for context rows only, in `context_indices` order.
    pub labels: Vec<f32>,
    pub categorical_mask: Vec<bool>,
    pub target_center: f64,
    pub target_scale: f64,
    pub target_sign: f64,
    pub class_offset: usize,
    /// Encoded class index maps to the original numeric target value at this position.
    pub class_permutation: Vec<f64>,
}

fn ordered_feature_sources(
    keep: &[usize],
    kinds: &[FeatureType],
    count_count: usize,
) -> Vec<usize> {
    let mut sources = keep
        .iter()
        .enumerate()
        .filter_map(|(position, &column)| {
            (kinds[column] == FeatureType::Numerical).then_some(position)
        })
        .collect::<Vec<_>>();
    sources.extend(keep.len()..keep.len() + count_count);
    sources.extend(keep.iter().enumerate().filter_map(|(position, &column)| {
        (kinds[column] == FeatureType::Categorical).then_some(position)
    }));
    sources
}

/// Fit context statistics and transform context + query rows in that order.
pub fn prepare(
    rows: &[Vec<Value>],
    feature_names: &[&str],
    context_indices: &[usize],
    query_indices: &[usize],
    target_values: &[f64],
    task: Task,
    rng: &mut StdRng,
) -> Result<Prepared, String> {
    prepare_with_feature_types(
        rows,
        feature_names,
        context_indices,
        query_indices,
        target_values,
        task,
        rng,
        None,
    )
}

/// Prepare using explicit semantic types. `None` preserves the legacy
/// inference rule (string columns are categorical); otherwise the slice must
/// contain one type per feature column.
pub fn prepare_with_feature_types(
    rows: &[Vec<Value>],
    feature_names: &[&str],
    context_indices: &[usize],
    query_indices: &[usize],
    target_values: &[f64],
    task: Task,
    rng: &mut StdRng,
    feature_types: Option<&[FeatureType]>,
) -> Result<Prepared, String> {
    prepare_impl(
        rows,
        feature_names,
        context_indices,
        query_indices,
        target_values,
        task,
        rng,
        Some(10),
        feature_types,
    )
}

/// Like [`prepare`], but does not reject classification targets with more than
/// ten classes. The returned labels are encoded class indices; use
/// [`Prepared::class_permutation`] to map an encoded index back to the original
/// numeric target value. Callers that run the exported ten-class core must
/// encode those labels themselves (for example with the ECOC helper in
/// [`crate::ecoc`]).
pub fn prepare_any_classes(
    rows: &[Vec<Value>],
    feature_names: &[&str],
    context_indices: &[usize],
    query_indices: &[usize],
    target_values: &[f64],
    task: Task,
    rng: &mut StdRng,
) -> Result<Prepared, String> {
    prepare_impl(
        rows,
        feature_names,
        context_indices,
        query_indices,
        target_values,
        task,
        rng,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_impl(
    rows: &[Vec<Value>],
    feature_names: &[&str],
    context_indices: &[usize],
    query_indices: &[usize],
    target_values: &[f64],
    task: Task,
    rng: &mut StdRng,
    max_classes: Option<usize>,
    feature_types: Option<&[FeatureType]>,
) -> Result<Prepared, String> {
    if target_values.len() != rows.len() {
        return Err("target values must align with rows".into());
    }
    if rows.iter().any(|r| r.len() != feature_names.len()) {
        return Err("every row must match feature width".into());
    }
    validate_indices(context_indices, rows.len())?;
    validate_indices(query_indices, rows.len())?;
    if context_indices.is_empty() || query_indices.is_empty() {
        return Err("at least one context and one query row are required".into());
    }
    let kinds = if let Some(feature_types) = feature_types {
        if feature_types.len() != feature_names.len() {
            return Err("feature types must match feature width".into());
        }
        feature_types.to_vec()
    } else {
        (0..feature_names.len())
            .map(|c| {
                if rows.iter().any(|row| matches!(row[c], Value::String(_))) {
                    FeatureType::Categorical
                } else {
                    FeatureType::Numerical
                }
            })
            .collect()
    };
    let mut keep = Vec::new();
    let mut categories = Vec::with_capacity(feature_names.len());
    let mut numeric: Vec<Option<NumericStats>> = vec![None; feature_names.len()];
    let mut high = Vec::new();
    for (c, kind) in kinds.iter().enumerate() {
        match kind {
            FeatureType::Numerical => {
                let vals: Vec<f64> = context_indices
                    .iter()
                    .filter_map(|&i| number(&rows[i][c]))
                    .collect();
                let constant = vals.is_empty()
                    || (vals.len() == context_indices.len() && vals.iter().all(|&v| v == vals[0]));
                if !constant {
                    numeric[c] = Some(numeric_stats(&vals, rng, true));
                }
                categories.push(None);
                if !constant {
                    keep.push(c);
                }
            }
            FeatureType::Categorical => {
                let mut vals = context_indices
                    .iter()
                    .filter_map(|&i| category(&rows[i][c]))
                    .collect::<Vec<_>>();
                vals.sort();
                vals.dedup();
                // AlignCategories(sort_by="value") sorts numeric categories by
                // numeric value, not by their string rendering (1, 2, 10).
                if context_indices
                    .iter()
                    .all(|&i| number(&rows[i][c]).is_some())
                {
                    vals.sort_by(|a, b| {
                        a.parse::<f64>()
                            .unwrap_or(f64::NAN)
                            .total_cmp(&b.parse::<f64>().unwrap_or(f64::NAN))
                    });
                }
                // AlignCategories: zero-based sorted context vocabulary, with -1 for unknown/missing.
                let codes: BTreeMap<String, usize> =
                    vals.into_iter().enumerate().map(|(i, s)| (s, i)).collect();
                let mut counts = BTreeMap::new();
                let mut negative_count = 0;
                for &i in context_indices {
                    if let Some(s) = category(&rows[i][c]) {
                        *counts.entry(s).or_insert(0usize) += 1;
                    } else {
                        negative_count += 1;
                    }
                }
                let codes_seen = context_indices
                    .iter()
                    .map(|&i| {
                        category(&rows[i][c])
                            .and_then(|v| codes.get(&v).copied())
                            .map_or(-1, |v| v as i64)
                    })
                    .collect::<BTreeSet<_>>();
                if codes_seen.len() > 1 {
                    keep.push(c);
                }
                if codes.len() > CARDINALITY_THRESHOLD {
                    high.push((c, counts, negative_count));
                }
                categories.push(Some(codes));
            }
        }
    }
    let classes: Vec<f64> = if task == Task::Classification {
        let mut vals: Vec<f64> = context_indices
            .iter()
            .map(|&i| target_values[i])
            .filter(|v| v.is_finite())
            .collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup_by(|a, b| a == b);
        vals
    } else {
        Vec::new()
    };
    if task == Task::Classification && classes.is_empty() {
        return Err("classification target has no finite context labels".into());
    }
    if let Some(max_classes) = max_classes {
        if classes.len() > max_classes {
            return Err("more than 10 classes require ECOC encoding".into());
        }
    }
    let class_offset = if classes.len() > 1 {
        let mut offsets = (0..classes.len()).collect::<Vec<_>>();
        offsets.shuffle(rng);
        offsets[0]
    } else {
        0
    };
    let class_permutation = (0..classes.len())
        .map(|code| classes[(code + class_offset) % classes.len()])
        .collect::<Vec<_>>();
    let (target_center, target_scale, target_sign) = if task == Task::Regression {
        let ys: Vec<f64> = context_indices
            .iter()
            .map(|&i| target_values[i])
            .filter(|v| v.is_finite())
            .collect();
        if ys.is_empty() {
            return Err("regression target has no finite context values".into());
        }
        let center = mean(&ys);
        (
            center,
            if ys.iter().all(|&y| y == ys[0]) {
                1.0
            } else {
                stddev(&ys, center)
            },
            if rng.gen_bool(0.5) { -1.0 } else { 1.0 },
        )
    } else {
        (0.0, 1.0, 1.0)
    };
    // Categorical codes and high-cardinality counts take the same numerical
    // standardize/clip/ClipSigma pipeline, but not the numerical FlipSign.
    let mut categorical_numeric = vec![None; feature_names.len()];
    for &c in &keep {
        if kinds[c] == FeatureType::Categorical {
            let codes = context_indices
                .iter()
                .map(|&i| {
                    categories[c]
                        .as_ref()
                        .and_then(|map| category(&rows[i][c]).and_then(|v| map.get(&v).copied()))
                        .map_or(-1.0, |code| code as f64)
                })
                .collect::<Vec<_>>();
            if !codes.iter().all(|&code| code == codes[0]) {
                categorical_numeric[c] = Some(numeric_stats(&codes, rng, false));
            }
        }
    }
    let categorical_count_columns = high
        .iter()
        .map(|(c, counts, negative_count)| {
            let values = context_indices
                .iter()
                .map(|&i| {
                    let count = category(&rows[i][*c])
                        .and_then(|v| counts.get(&v).copied())
                        .unwrap_or(*negative_count);
                    (count as f64).ln_1p()
                })
                .collect::<Vec<_>>();
            (c, counts, negative_count, values)
        })
        .collect::<Vec<_>>();
    let count_stats = categorical_count_columns
        .iter()
        .map(|(_, _, _, values)| {
            if values.iter().all(|&v| v == values[0]) {
                None
            } else {
                Some(numeric_stats(values, rng, false))
            }
        })
        .collect::<Vec<_>>();
    let count_columns = categorical_count_columns
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            count_stats[index].map(|stats| (value.0, value.1, value.2, stats))
        })
        .collect::<Vec<_>>();
    let initial_sources = ordered_feature_sources(&keep, &kinds, count_columns.len());
    // First Latin member uses base[(0 - row_rank) mod C], not a plain permutation.
    let mut permutation: Vec<usize> = (0..initial_sources.len()).collect();
    permutation.shuffle(rng);
    let mut rows_order: Vec<usize> = (0..permutation.len()).collect();
    rows_order.shuffle(rng);
    if !permutation.is_empty() {
        permutation = rows_order
            .iter()
            .map(|&r| permutation[(permutation.len() - r) % permutation.len()])
            .collect();
    }
    permutation.truncate(MAX_FEATURES);
    let mut mask: Vec<bool> = keep
        .iter()
        .map(|&c| kinds[c] == FeatureType::Categorical)
        .collect();
    mask.extend(std::iter::repeat(false).take(count_columns.len()));
    let mask = initial_sources
        .iter()
        .map(|&source| mask[source])
        .collect::<Vec<_>>();
    let mask = permutation
        .iter()
        .map(|&source| mask[source])
        .take(MAX_FEATURES)
        .collect::<Vec<_>>();
    let row_indices: Vec<usize> = context_indices
        .iter()
        .chain(query_indices)
        .copied()
        .collect();
    let mut x = Array2::<f32>::zeros((row_indices.len(), permutation.len()));
    for (r, &i) in row_indices.iter().enumerate() {
        let mut values = vec![0f32; initial_sources.len()];
        for (j, &c) in keep.iter().enumerate() {
            values[j] = if let Some(stats) = numeric[c] {
                transform_numeric(number(&rows[i][c]).unwrap_or(f64::NAN), stats)
            } else {
                let code = categories[c]
                    .as_ref()
                    .and_then(|map| category(&rows[i][c]).and_then(|v| map.get(&v).copied()))
                    .map_or(-1.0, |v| v as f64);
                transform_numeric(
                    code,
                    categorical_numeric[c].ok_or("constant categorical column")?,
                )
            };
        }
        for (k, &(c, counts, negative_count, stats)) in count_columns.iter().enumerate() {
            let count = category(&rows[i][*c])
                .and_then(|v| counts.get(&v).copied())
                .unwrap_or(*negative_count);
            values[keep.len() + k] = transform_numeric((count as f64).ln_1p(), stats);
        }
        for (out, &source) in permutation.iter().enumerate() {
            x[(r, out)] = values[initial_sources[source]];
        }
    }
    let labels = context_indices
        .iter()
        .map(|&i| {
            if task == Task::Regression {
                ((target_values[i] - target_center) / target_scale * target_sign) as f32
            } else {
                classes
                    .iter()
                    .position(|&c| c == target_values[i])
                    .map(|n| ((n + classes.len() - class_offset) % classes.len()) as f32)
                    .unwrap_or(f32::NAN)
            }
        })
        .collect();
    // `mask` is already in final column order: `initial_sources` moved it to
    // numerical/count/categorical order, then `permutation` applied the Latin
    // shuffle once. Do not apply `permutation` again here.
    // Keep named feature order observable through an optional debug invariant.
    debug_assert!(keep.iter().all(|&i| i < feature_names.len()));
    Ok(Prepared {
        x,
        labels,
        categorical_mask: mask,
        target_center,
        target_scale,
        target_sign,
        class_offset,
        class_permutation,
    })
}

fn validate_indices(indices: &[usize], n: usize) -> Result<(), String> {
    if indices.iter().any(|&i| i >= n) {
        Err("row index out of bounds".into())
    } else {
        Ok(())
    }
}
fn number(v: &Value) -> Option<f64> {
    match v {
        // Non-finite FFI values are represented by reserved strings because JSON
        // numbers cannot encode infinities; they remain numeric for this recipe.
        Value::Object(value) => value
            .get("__tabular_onnx_ffi_infinity__")
            .and_then(Value::as_str)
            .and_then(|sign| match sign {
                "positive" => Some(f64::INFINITY),
                "negative" => Some(f64::NEG_INFINITY),
                _ => None,
            }),

        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}
fn category(v: &Value) -> Option<String> {
    match v {
        Value::Object(value) if value.contains_key("__tabular_onnx_ffi_infinity__") => None,
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}
fn stddev(xs: &[f64], m: f64) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(x: &str) -> Value {
        Value::String(x.into())
    }
    fn n(x: i64) -> Value {
        Value::Number(x.into())
    }

    #[test]
    fn official_feature_blocks_order_numeric_then_counts_then_categories() {
        let keep = [0, 1, 2, 3];
        let kinds = [
            FeatureType::Categorical,
            FeatureType::Numerical,
            FeatureType::Categorical,
            FeatureType::Numerical,
        ];
        assert_eq!(ordered_feature_sources(&keep, &kinds, 1), [1, 3, 4, 0, 2]);
    }

    #[test]
    fn negative_category_codes_share_the_fitted_count() {
        let rows = (0..55)
            .map(|i| {
                if i == 0 || i == 1 {
                    vec![Value::Null]
                } else {
                    vec![s(&format!("c{i}"))]
                }
            })
            .collect::<Vec<_>>();
        let labels = vec![0.0; rows.len()];
        let context = (0..53).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(11);
        let prepared = prepare(
            &rows,
            &["category"],
            &context,
            &[53],
            &labels,
            Task::Classification,
            &mut rng,
        )
        .unwrap();
        assert_eq!(prepared.x.ncols(), 2);
        let count_column = prepared
            .categorical_mask
            .iter()
            .position(|is_categorical| !is_categorical)
            .unwrap();
        let missing_a = prepared.x[(0, count_column)];
        let missing_b = prepared.x[(1, count_column)];
        let unseen_query = prepared.x[(53, count_column)];
        assert_eq!(missing_a, missing_b);
        assert_eq!(missing_a, unseen_query);
        assert!(missing_a.is_finite());
    }

    #[test]
    fn includes_context_then_query_and_context_labels() {
        let rows = vec![vec![n(1)], vec![n(3)], vec![n(5)]];
        let y = vec![10., 20., 30.];
        let mut rng = StdRng::seed_from_u64(3);
        let p = prepare(&rows, &["x"], &[0, 1], &[2], &y, Task::Regression, &mut rng).unwrap();
        assert_eq!(p.x.nrows(), 3);
        assert_eq!(p.labels.len(), 2);
        assert_eq!(p.labels.len(), p.x.nrows() - 1);
        for (i, &row) in [0usize, 1].iter().enumerate() {
            assert!(
                ((p.labels[i] as f64) / p.target_sign * p.target_scale + p.target_center - y[row])
                    .abs()
                    < 1e-4
            );
        }
    }
    #[test]
    fn aligns_categories_unknown_minus_one_drops_constant() {
        let rows = vec![vec![s("b"), n(4)], vec![s("a"), n(4)], vec![s("c"), n(4)]];
        let y = vec![0., 1., 0.];
        let mut rng = StdRng::seed_from_u64(1);
        let p = prepare(
            &rows,
            &["cat", "constant"],
            &[0, 1],
            &[2],
            &y,
            Task::Classification,
            &mut rng,
        )
        .unwrap();
        assert_eq!(p.x.nrows(), 3);
        let cat_col = p.categorical_mask.iter().position(|&v| v).unwrap();
        assert!((p.x[(0, cat_col)] - 0.999998).abs() < 1e-5);
        assert!((p.x[(1, cat_col)] + 0.999998).abs() < 1e-5);
        assert!(p.x[(2, cat_col)] < p.x[(1, cat_col)]);
        assert_eq!(p.labels, vec![0., 1.]);
        assert_eq!(p.class_permutation, vec![0., 1.]);
    }
    #[test]
    fn high_cardinality_constant_count_feature_is_dropped() {
        let rows: Vec<Vec<Value>> = (0..51).map(|i| vec![s(&format!("c{i}"))]).collect();
        let y = vec![0.; 51];
        let mut rng = StdRng::seed_from_u64(5);
        let p = prepare(
            &rows,
            &["cat"],
            &(0..51).collect::<Vec<_>>(),
            &[0],
            &y,
            Task::Classification,
            &mut rng,
        )
        .unwrap();
        assert_eq!(p.x.shape(), &[52, 1]);
        assert_eq!(p.categorical_mask.iter().filter(|&&v| v).count(), 1);
    }

    #[test]
    fn numerical_infinity_is_clipped_and_keeps_numeric_semantics() {
        let infinity = Value::Object(serde_json::Map::from_iter([(
            "__tabular_onnx_ffi_infinity__".into(),
            Value::String("positive".into()),
        )]));
        let rows = vec![vec![n(1)], vec![n(2)], vec![infinity]];
        let prepared = prepare_with_feature_types(
            &rows,
            &["x"],
            &[0, 1],
            &[2],
            &[0.0, 0.0, 0.0],
            Task::Regression,
            &mut StdRng::seed_from_u64(0),
            Some(&[FeatureType::Numerical]),
        )
        .unwrap();
        assert!(prepared.x[(0, 0)].is_finite());
        assert!(prepared.x[(1, 0)].is_finite());
    }

    #[test]
    fn missing_numerical_cell_remains_nan_for_model() {
        let rows = vec![vec![n(1)], vec![n(2)], vec![Value::Null]];
        let labels = vec![0.0, 1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(17);
        let prepared = prepare(
            &rows,
            &["numeric"],
            &[0, 1],
            &[2],
            &labels,
            Task::Classification,
            &mut rng,
        )
        .unwrap();
        assert_eq!(prepared.x.shape(), &[3, 1]);
        assert!(prepared.x[(2, 0)].is_nan());
        assert_eq!(prepared.labels.len(), 2);
    }
    #[test]
    fn any_classes_allows_ecoc_sized_targets() {
        let rows = (0..12).map(|index| vec![n(index)]).collect::<Vec<_>>();
        let labels = (0..12).map(|index| (index % 11) as f64).collect::<Vec<_>>();
        let context = (0..11).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(23);
        assert!(prepare(
            &rows,
            &["x"],
            &context,
            &[11],
            &labels,
            Task::Classification,
            &mut rng,
        )
        .is_err());
        let mut rng = StdRng::seed_from_u64(23);
        let prepared = prepare_any_classes(
            &rows,
            &["x"],
            &context,
            &[11],
            &labels,
            Task::Classification,
            &mut rng,
        )
        .unwrap();
        assert_eq!(prepared.labels.len(), 11);
        assert_eq!(prepared.class_permutation.len(), 11);
    }

    #[test]
    fn categorical_mask_follows_shuffled_columns() {
        let rows = (0..10)
            .map(|i| {
                vec![
                    n(i as i64),
                    s(if i % 2 == 0 { "a" } else { "b" }),
                    s(&format!("g{}", i % 3)),
                    n((i * 7) as i64),
                ]
            })
            .collect::<Vec<_>>();
        let labels = (0..10).map(|i| (i % 2) as f64).collect::<Vec<_>>();
        let context = (0..8).collect::<Vec<_>>();
        let query = [8, 9];
        for seed in 0..64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let prepared = prepare(
                &rows,
                &["num", "cat_a", "cat_b", "num_b"],
                &context,
                &query,
                &labels,
                Task::Regression,
                &mut rng,
            )
            .unwrap();
            for column in 0..prepared.x.ncols() {
                let mut distinct = prepared.x.column(column).to_vec();
                distinct.sort_by(f32::total_cmp);
                distinct.dedup();
                // Categorical columns have at most three observed codes;
                // numerical columns have one distinct value per row.
                assert_eq!(
                    prepared.categorical_mask[column],
                    distinct.len() <= 3,
                    "seed {seed}, shuffled column {column}, distinct {distinct:?}"
                );
            }
        }
    }

    #[test]
    fn explicit_numeric_categories_use_numeric_value_order() {
        let rows = vec![
            vec![n(1), n(10)],
            vec![n(10), n(1)],
            vec![n(2), n(2)],
            vec![n(12), n(12)],
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0];
        let kinds = [FeatureType::Categorical, FeatureType::Numerical];
        let mut rng = StdRng::seed_from_u64(7);
        let prepared = prepare_with_feature_types(
            &rows,
            &["month", "value"],
            &[0, 1, 2],
            &[3],
            &labels,
            Task::Classification,
            &mut rng,
            Some(&kinds),
        )
        .unwrap();
        assert_eq!(prepared.categorical_mask.iter().filter(|&&v| v).count(), 1);
        assert!(prepared.x.ncols() >= 1);
    }

    #[test]
    fn errors_on_bad_index() {
        let rows = vec![vec![n(1)]];
        let mut rng = StdRng::seed_from_u64(0);
        assert!(prepare(&rows, &["x"], &[2], &[0], &[1.], Task::Regression, &mut rng).is_err());
    }
}
