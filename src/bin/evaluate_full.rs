//! Full-feature Kumo evaluation using the dynamic FP16 core and the Rust recipe.

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};
use std::{error::Error, fs, path::Path, process::Command, time::Instant};
use tabular_onnx::{
    kumo::KumoBatch,
    kumo_recipe::{prepare, Task as RecipeTask},
    OnnxSession, Task,
};

type EvalResult<T> = Result<T, Box<dyn Error>>;
type Dataset = (Vec<String>, Vec<Vec<Value>>, Vec<f64>);
type BatchIndices = (Vec<usize>, Vec<usize>);
const DEFAULT_CONTEXT: usize = 32;
const QUERIES: usize = 2;

fn load_dataset(path: &str, target: &str) -> EvalResult<Dataset> {
    let command = Command::new("duckdb")
        .args([
            "-json",
            "-c",
            &format!("SELECT * FROM read_parquet('{}')", path.replace('\'', "''")),
        ])
        .output()?;
    if !command.status.success() {
        return Err(format!("DuckDB: {}", String::from_utf8_lossy(&command.stderr)).into());
    }
    let rows: Vec<serde_json::Map<String, Value>> = serde_json::from_slice(&command.stdout)?;
    let description = Command::new("duckdb")
        .args([
            "-json",
            "-c",
            &format!("DESCRIBE SELECT * FROM read_parquet('{}')", path),
        ])
        .output()?;
    if !description.status.success() {
        return Err(format!("DuckDB: {}", String::from_utf8_lossy(&description.stderr)).into());
    }
    let schema: Vec<Value> = serde_json::from_slice(&description.stdout)?;
    let names = schema
        .iter()
        .filter_map(|col| col["column_name"].as_str())
        .filter(|name| *name != target)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if names.is_empty() || rows.len() < DEFAULT_CONTEXT + QUERIES {
        return Err("not enough features or dataset rows".into());
    }
    let mut features = Vec::with_capacity(rows.len());
    let mut labels = Vec::with_capacity(rows.len());
    for row in rows {
        features.push(
            names
                .iter()
                .map(|name| row.get(name).cloned().unwrap_or(Value::Null))
                .collect(),
        );
        let y = if target == "bad_customer" {
            match row.get(target).and_then(Value::as_str) {
                Some("No") => 0.0,
                Some("Yes") => 1.0,
                _ => return Err("unexpected classification target".into()),
            }
        } else {
            row.get(target)
                .and_then(Value::as_f64)
                .ok_or("invalid regression target")?
        };
        if !y.is_finite() {
            return Err("non-finite target".into());
        }
        labels.push(y);
    }
    Ok((names, features, labels))
}

fn split(labels: &[f64], classification: bool, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut train = Vec::new();
    let mut test = Vec::new();
    let groups = if classification {
        vec![Some(0.0), Some(1.0)]
    } else {
        vec![None]
    };
    for group in groups {
        let mut indices = (0..labels.len())
            .filter(|&i| group.map_or(true, |g| labels[i] == g))
            .collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        let count = (indices.len() as f64 * 0.2).round() as usize;
        test.extend_from_slice(&indices[..count]);
        train.extend_from_slice(&indices[count..]);
    }
    test.shuffle(&mut rng);
    let mut val_rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A_5A5A_A5A5);
    let mut fit = Vec::new();
    let mut validation = Vec::new();
    let groups = if classification {
        vec![Some(0.0), Some(1.0)]
    } else {
        vec![None]
    };
    for group in groups {
        let mut indices = train
            .iter()
            .copied()
            .filter(|&i| group.map_or(true, |g| labels[i] == g))
            .collect::<Vec<_>>();
        indices.shuffle(&mut val_rng);
        let count = (indices.len() as f64 * 0.2).round() as usize;
        validation.extend_from_slice(&indices[..count]);
        fit.extend_from_slice(&indices[count..]);
    }
    validation.shuffle(&mut val_rng);
    (fit, validation, test)
}

#[allow(clippy::too_many_arguments)]
fn infer(
    session: &mut OnnxSession,
    features: &[Vec<Value>],
    names: &[&str],
    labels: &[f64],
    pool: &[usize],
    queries: &[usize],
    task: Task,
    seed: u64,
    context_rows: usize,
    trace_batches: Option<&mut Vec<Value>>,
) -> EvalResult<(Vec<f64>, f64, Vec<BatchIndices>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = Instant::now();
    let mut predictions = Vec::with_capacity(queries.len());
    let mut batches = Vec::new();
    let mut trace_batches = trace_batches;
    for query in queries.chunks(QUERIES) {
        let mut context = if task == Task::Classification {
            [0.0, 1.0]
                .iter()
                .flat_map(|&target| {
                    pool.iter()
                        .copied()
                        .filter(|&i| labels[i] == target)
                        .collect::<Vec<_>>()
                        .choose_multiple(&mut rng, context_rows / 2)
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            pool.choose_multiple(&mut rng, context_rows)
                .copied()
                .collect::<Vec<_>>()
        };
        if context.len() != context_rows {
            return Err("insufficient context rows".into());
        }
        context.shuffle(&mut rng);
        batches.push((context.clone(), query.to_vec()));
        let recipe_task = if task == Task::Classification {
            RecipeTask::Classification
        } else {
            RecipeTask::Regression
        };
        let prepared = prepare(
            features,
            names,
            &context,
            query,
            labels,
            recipe_task,
            &mut rng,
        )?;
        let batch = KumoBatch::from_dynamic_features(
            &prepared.x,
            &prepared.labels,
            task,
            &prepared.categorical_mask,
        )?;
        let output = session.run_kumo(batch)?;
        if let Some(trace) = trace_batches.as_deref_mut() {
            trace.push(json!({
                "context_indices": context,
                "query_indices": query,
                "x": prepared.x.rows().into_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
                "y": prepared.labels,
                "categorical_mask": prepared.categorical_mask,
                "class_permutation": prepared.class_permutation,
                "target_center": prepared.target_center,
                "target_scale": prepared.target_scale,
                "target_sign": prepared.target_sign,
                "rust_output": output.rows().into_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
            }));
        }
        if output.nrows() != query.len()
            || output.ncols()
                != if task == Task::Classification {
                    10
                } else {
                    999
                }
            || output.iter().any(|value| !value.is_finite())
        {
            return Err(
                format!("unexpected or non-finite output shape {:?}", output.shape()).into(),
            );
        }
        for row in output.rows() {
            if task == Task::Classification {
                // Undo category shift before comparing the two original classes.
                let no = prepared
                    .class_permutation
                    .iter()
                    .position(|&v| v == 0.0)
                    .ok_or("missing No class")?;
                let yes = prepared
                    .class_permutation
                    .iter()
                    .position(|&v| v == 1.0)
                    .ok_or("missing Yes class")?;
                predictions.push((row[yes] - row[no]) as f64);
            } else {
                let mut quantiles = row.iter().copied().collect::<Vec<_>>();
                quantiles.sort_by(f32::total_cmp);
                predictions.push(
                    quantiles[499] as f64 * prepared.target_scale * prepared.target_sign
                        + prepared.target_center,
                );
            }
        }
    }
    Ok((predictions, start.elapsed().as_secs_f64(), batches))
}

fn counts(truth: &[f64], scores: &[f64], threshold: f64) -> [usize; 4] {
    let mut c = [0; 4];
    for (&y, &p) in truth.iter().zip(scores) {
        c[2 * (y == 1.0) as usize + (p > threshold) as usize] += 1;
    }
    c
}

fn balanced(truth: &[f64], scores: &[f64], threshold: f64) -> f64 {
    let [tn, fp, fn_, tp] = counts(truth, scores, threshold);
    (tn as f64 / (tn + fp) as f64 + tp as f64 / (tp + fn_) as f64) / 2.0
}

fn threshold(truth: &[f64], scores: &[f64]) -> f64 {
    let mut candidates = scores.to_vec();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    candidates
        .into_iter()
        .max_by(|a, b| {
            balanced(truth, scores, *a)
                .total_cmp(&balanced(truth, scores, *b))
                .then_with(|| a.total_cmp(b))
        })
        .unwrap()
}

fn auc(truth: &[f64], scores: &[f64]) -> f64 {
    let mut pairs = truth
        .iter()
        .zip(scores)
        .map(|(&y, &p)| (p, y))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let positive = truth.iter().filter(|&&y| y == 1.0).count() as f64;
    let negative = truth.len() as f64 - positive;
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        rank_sum += ((i + j + 1) as f64 / 2.0)
            * pairs[i..j].iter().filter(|(_, y)| *y == 1.0).count() as f64;
        i = j;
    }
    (rank_sum - positive * (positive + 1.0) / 2.0) / (positive * negative)
}

fn evaluate(
    size: &str,
    task: Task,
    seed: u64,
    context_rows: usize,
    trace: bool,
) -> EvalResult<Value> {
    let classification = task == Task::Classification;
    let (path, target) = if classification {
        (
            "dataset/classification/bad_customer_detection.parquet",
            "bad_customer",
        )
    } else {
        (
            "dataset/regression/airfoil_self_noise.parquet",
            "scaled-sound-pressure",
        )
    };
    let model = format!(
        "{size}/kumo-tabular-{size}-{}-dynamic-fp16.onnx",
        if classification {
            "classification"
        } else {
            "regression"
        }
    );
    if !Path::new(&model).is_file() {
        return Err(format!(
            "missing new dynamic export: {model}; run updated GitHub Action first"
        )
        .into());
    }
    let (names, features, labels) = load_dataset(path, target)?;
    let names_ref = names.iter().map(String::as_str).collect::<Vec<_>>();
    let (fit, validation, test) = split(&labels, classification, seed);
    if fit.len() < context_rows
        || (classification
            && [0.0, 1.0].iter().any(|&label| {
                fit.iter().filter(|&&index| labels[index] == label).count() < context_rows / 2
            }))
    {
        return Err("insufficient fit rows for requested context size".into());
    }
    let mut session = OnnxSession::load(&model)?;
    if session.input_names() != ["x", "y", "categorical_mask"]
        || session.output_names() != ["predictions"]
    {
        return Err("unexpected ONNX tensor contract".into());
    }
    let mut validation_prepared = Vec::new();
    let (validation_scores, _, validation_batches) = infer(
        &mut session,
        &features,
        &names_ref,
        &labels,
        &fit,
        &validation,
        task,
        seed,
        context_rows,
        trace.then_some(&mut validation_prepared),
    )?;
    let mut test_prepared = Vec::new();
    let (predictions, elapsed, test_batches) = infer(
        &mut session,
        &features,
        &names_ref,
        &labels,
        &fit,
        &test,
        task,
        seed,
        context_rows,
        trace.then_some(&mut test_prepared),
    )?;
    let truth = test.iter().map(|&i| labels[i]).collect::<Vec<_>>();
    let validation_truth = validation.iter().map(|&i| labels[i]).collect::<Vec<_>>();
    if trace {
        fs::create_dir_all("output")?;
        let trace_path = format!(
            "output/{size}-{}-context-{context_rows}-seed-{seed}-trace.json",
            if classification {
                "classification"
            } else {
                "regression"
            }
        );
        let trace_json = json!({
            "dataset": path, "target": target, "seed": seed, "context_rows": context_rows,
            "features": names, "fit": fit, "validation": validation, "test": test,
            "validation_batches": validation_batches.iter().map(|(context, query)| json!({"context": context, "query": query})).collect::<Vec<_>>(),
            "test_batches": test_batches.iter().map(|(context, query)| json!({"context": context, "query": query})).collect::<Vec<_>>(),
            "validation_scores": validation_scores, "test_scores": predictions,
            "validation_prepared": validation_prepared, "test_prepared": test_prepared,
        });
        fs::write(
            &trace_path,
            format!("{}\n", serde_json::to_string_pretty(&trace_json)?),
        )?;
    }
    let metrics;
    let cut = if classification {
        Some(threshold(&validation_truth, &validation_scores))
    } else {
        None
    };
    if let Some(cut) = cut {
        let [tn, fp, fn_, tp] = counts(&truth, &predictions, cut);
        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / (tp + fn_).max(1) as f64;
        metrics = json!({"accuracy": (tn + tp) as f64 / truth.len() as f64,
            "balanced_accuracy": balanced(&truth, &predictions, cut),
            "f1_yes": 2.0 * precision * recall / (precision + recall).max(f64::EPSILON),
            "precision_yes": precision, "recall_yes": recall, "roc_auc_yes": auc(&truth, &predictions),
            "confusion_matrix_no_yes": [[tn, fp], [fn_, tp]],
            "majority_no_accuracy": (tn + fp) as f64 / truth.len() as f64});
    } else {
        let mean = truth.iter().sum::<f64>() / truth.len() as f64;
        let train_mean = fit.iter().map(|&i| labels[i]).sum::<f64>() / fit.len() as f64;
        let sq = truth
            .iter()
            .zip(&predictions)
            .map(|(y, p)| (y - p).powi(2))
            .sum::<f64>();
        metrics = json!({"mae": truth.iter().zip(&predictions).map(|(y,p)| (y-p).abs()).sum::<f64>() / truth.len() as f64,
            "rmse": (sq/truth.len() as f64).sqrt(),
            "r2": 1.0-sq/truth.iter().map(|y| (y-mean).powi(2)).sum::<f64>(),
            "fit_mean_baseline_mae": truth.iter().map(|y| (y-train_mean).abs()).sum::<f64>() / truth.len() as f64});
    }
    Ok(
        json!({"task": if classification { "classification" } else { "regression" },
        "engine": "rust-ort", "pipeline": "kumo-default-single-estimator", "precision": "fp16",
        "model": model, "size": size, "dataset": path, "features": names, "target": target,
        "context_rows_per_batch": context_rows, "queries_per_batch": QUERIES,
        "context_strategy": "class-balanced-random" , "seed": seed, "fit_rows": fit.len(),
        "validation_rows": validation.len(), "test_rows": test.len(),
        "classification_threshold": cut, "prediction_seconds": elapsed, "metrics": metrics}),
    )
}

fn main() -> EvalResult<()> {
    let mut args = std::env::args().skip(1);
    let mut seed = 42;
    let mut size = "both".to_owned();
    let mut context_rows = DEFAULT_CONTEXT;
    let mut trace = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => seed = args.next().ok_or("missing --seed value")?.parse()?,
            "--size" => size = args.next().ok_or("missing --size value")?,
            "--context-rows" => {
                context_rows = args.next().ok_or("missing --context-rows value")?.parse()?
            }
            "--trace" => trace = true,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if context_rows < 2 || context_rows % 2 != 0 {
        return Err("--context-rows must be an even integer of at least 2".into());
    }
    let sizes = match size.as_str() {
        "both" => vec!["small", "medium"],
        "small" => vec!["small"],
        "medium" => vec!["medium"],
        _ => return Err("size must be small, medium or both".into()),
    };
    for size in sizes {
        let report = vec![
            evaluate(size, Task::Classification, seed, context_rows, trace)?,
            evaluate(size, Task::Regression, seed, context_rows, trace)?,
        ];
        let path = if context_rows == DEFAULT_CONTEXT {
            format!("output/{size}-full-evaluation-rust.json")
        } else {
            format!("output/{size}-full-context-{context_rows}-evaluation-rust.json")
        };
        fs::create_dir_all("output")?;
        fs::write(
            &path,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        println!(
            "{size} report: {path}\n{}",
            serde_json::to_string_pretty(&report)?
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_out_split_has_no_overlap() {
        let labels = (0..100)
            .map(|i| (i % 10 == 0) as u8 as f64)
            .collect::<Vec<_>>();
        let (fit, validation, test) = split(&labels, true, 42);
        assert_eq!(fit.len() + validation.len() + test.len(), 100);
        assert!(fit
            .iter()
            .all(|i| !validation.contains(i) && !test.contains(i)));
        assert!(validation.iter().all(|i| !test.contains(i)));
    }

    #[test]
    fn auc_and_threshold_handle_ties() {
        let truth = [0.0, 1.0, 0.0, 1.0];
        let scores = [0.0, 0.0, 1.0, 1.0];
        assert!((auc(&truth, &scores) - 0.5).abs() < 1e-12);
        assert!(balanced(&truth, &scores, threshold(&truth, &scores)) >= 0.5);
    }
}
