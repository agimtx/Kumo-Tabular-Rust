//! Dataset evaluation for the fixed-shape small and medium Kumo ONNX exports.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use tabular_onnx::kumo::KumoBatch;
use tabular_onnx::{OnnxSession, Task};

type EvalResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelSize {
    Small,
    Medium,
}

impl ModelSize {
    fn parse(value: &str) -> EvalResult<Self> {
        match value {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            _ => Err(format!("unknown size: {value}; expected small, medium or both").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
        }
    }
}

impl ModelSize {
    fn model_path(self, task: Task) -> &'static str {
        match (self, task) {
            (Self::Small, Task::Classification) => {
                "small/kumo-tabular-small-classification-fp16.onnx"
            }
            (Self::Small, Task::Regression) => "small/kumo-tabular-small-regression-fp16.onnx",
            (Self::Medium, Task::Classification) => {
                "medium/kumo-tabular-medium-classification-fp16.onnx"
            }
            (Self::Medium, Task::Regression) => "medium/kumo-tabular-medium-regression-fp16.onnx",
        }
    }
}

struct Dataset {
    features: Vec<[f64; 3]>,
    targets: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextSelection {
    Random,
    Nearest,
}

impl ContextSelection {
    fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Nearest => "nearest",
        }
    }
}

struct Inference {
    predictions: Vec<f64>,
    scores: Vec<f64>,
    seconds: f64,
}

fn load_data(
    path: &str,
    features: [&str; 3],
    target: &str,
    classification: bool,
) -> EvalResult<Dataset> {
    let quote = |name: &str| format!("\"{}\"", name.replace('"', "\"\""));
    let columns = features
        .iter()
        .copied()
        .chain(std::iter::once(target))
        .map(quote)
        .collect::<Vec<_>>();
    let query = format!(
        "SELECT {} FROM read_parquet('{}')",
        columns.join(", "),
        path.replace('\'', "''")
    );
    let output = Command::new("duckdb")
        .args(["-json", "-c", &query])
        .output()?;
    if !output.status.success() {
        return Err(format!("DuckDB failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    let rows: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    if rows.len() < 6 {
        return Err("dataset must contain at least six rows".into());
    }
    let mut data = Dataset {
        features: Vec::with_capacity(rows.len()),
        targets: Vec::with_capacity(rows.len()),
    };
    for row in rows {
        let mut values = [0.0; 3];
        for (col, name) in features.iter().enumerate() {
            values[col] = row[*name]
                .as_f64()
                .ok_or_else(|| format!("missing/non-numeric feature {name}"))?;
            if !values[col].is_finite() {
                return Err(format!("non-finite feature {name}").into());
            }
        }
        let label = if classification {
            match row[target].as_str() {
                Some("No") => 0.0,
                Some("Yes") => 1.0,
                _ => return Err("classification target must be No or Yes".into()),
            }
        } else {
            row[target]
                .as_f64()
                .ok_or("missing/non-numeric regression target")?
        };
        if !label.is_finite() {
            return Err("non-finite target".into());
        }
        data.features.push(values);
        data.targets.push(label);
    }
    Ok(data)
}

fn split(data: &Dataset, classification: bool, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    if classification {
        for label in [0.0, 1.0] {
            let mut indices = (0..data.targets.len())
                .filter(|&i| data.targets[i] == label)
                .collect::<Vec<_>>();
            indices.shuffle(rng);
            let count = (indices.len() as f64 * 0.2).round() as usize;
            test.extend_from_slice(&indices[..count]);
            train.extend_from_slice(&indices[count..]);
        }
    } else {
        let mut indices = (0..data.targets.len()).collect::<Vec<_>>();
        indices.shuffle(rng);
        let count = (indices.len() as f64 * 0.2).round() as usize;
        test.extend_from_slice(&indices[..count]);
        train.extend_from_slice(&indices[count..]);
    }
    test.shuffle(rng);
    (train, test)
}

fn standardize(data: &Dataset, train: &[usize]) -> Vec<[f32; 3]> {
    let mut mean = [0.0; 3];
    let mut variance = [0.0; 3];
    for &i in train {
        for col in 0..3 {
            mean[col] += data.features[i][col] / train.len() as f64;
        }
    }
    for &i in train {
        for col in 0..3 {
            variance[col] += (data.features[i][col] - mean[col]).powi(2) / train.len() as f64;
        }
    }
    data.features
        .iter()
        .map(|row| {
            std::array::from_fn(|col| {
                let scale = variance[col].sqrt().max(1e-6);
                ((row[col] - mean[col]) / scale).clamp(-100.0, 100.0) as f32
            })
        })
        .collect()
}

fn auc(labels: &[f64], scores: &[f64]) -> f64 {
    let mut ranked = scores
        .iter()
        .copied()
        .zip(labels.iter().copied())
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let positives = labels.iter().filter(|&&y| y == 1.0).count() as f64;
    let negatives = labels.len() as f64 - positives;
    let mut sum_positive_ranks = 0.0;
    let mut start = 0;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len() && ranked[start].0 == ranked[end].0 {
            end += 1;
        }
        let average_rank = (start + end + 1) as f64 / 2.0;
        sum_positive_ranks +=
            ranked[start..end].iter().filter(|(_, y)| *y == 1.0).count() as f64 * average_rank;
        start = end;
    }
    (sum_positive_ranks - positives * (positives + 1.0) / 2.0) / (positives * negatives)
}

fn choose_context(
    pool: &[usize],
    query: &[usize],
    features: &[[f32; 3]],
    targets: &[f64],
    task: Task,
    selection: ContextSelection,
    rng: &mut StdRng,
) -> Vec<usize> {
    let nearest = |candidates: &[usize], count: usize| {
        let mut ranked = candidates
            .iter()
            .copied()
            .map(|index| {
                let distance = query
                    .iter()
                    .map(|&q| {
                        (0..3)
                            .map(|col| (features[index][col] - features[q][col]).powi(2))
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / query.len() as f32;
                (index, distance)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(count)
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    };
    let mut context = if task == Task::Classification {
        [0.0, 1.0]
            .iter()
            .flat_map(|&label| {
                let candidates = pool
                    .iter()
                    .copied()
                    .filter(|&index| targets[index] == label)
                    .collect::<Vec<_>>();
                match selection {
                    ContextSelection::Random => {
                        candidates.choose_multiple(rng, 2).copied().collect()
                    }
                    ContextSelection::Nearest => nearest(&candidates, 2),
                }
            })
            .collect::<Vec<_>>()
    } else {
        match selection {
            ContextSelection::Random => pool.choose_multiple(rng, 4).copied().collect(),
            ContextSelection::Nearest => nearest(pool, 4),
        }
    };
    context.shuffle(rng);
    context
}

fn infer(
    session: &mut OnnxSession,
    data: &Dataset,
    features: &[[f32; 3]],
    pool: &[usize],
    queries: &[usize],
    task: Task,
    selection: ContextSelection,
    seed: u64,
) -> EvalResult<Inference> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut predictions = Vec::with_capacity(queries.len());
    let mut scores = Vec::with_capacity(queries.len());
    let start = Instant::now();
    for query in queries.chunks(2) {
        let context = choose_context(
            pool,
            query,
            features,
            &data.targets,
            task,
            selection,
            &mut rng,
        );
        if context.len() != 4 {
            return Err("not enough context rows".into());
        }
        let center = if task == Task::Classification {
            0.0
        } else {
            context.iter().map(|&i| data.targets[i]).sum::<f64>() / 4.0
        };
        let scale = if task == Task::Classification {
            1.0
        } else {
            let std = (context
                .iter()
                .map(|&i| (data.targets[i] - center).powi(2))
                .sum::<f64>()
                / 4.0)
                .sqrt();
            if std == 0.0 {
                1.0
            } else {
                std
            }
        };
        let labels = context
            .iter()
            .map(|&i| ((data.targets[i] - center) / scale) as f32)
            .collect::<Vec<_>>();
        let mut indices = context;
        indices.extend_from_slice(query);
        if query.len() == 1 {
            indices.push(query[0]);
        }
        let matrix = Array2::from_shape_fn((6, 3), |(row, col)| features[indices[row]][col]);
        let batch = KumoBatch::from_features(&matrix, &labels, task, &[false; 3])?;
        let output = session.run_kumo(batch)?;
        let expected_width = if task == Task::Classification {
            10
        } else {
            999
        };
        if output.shape() != [2, expected_width] || output.iter().any(|value| !value.is_finite()) {
            return Err("unexpected or non-finite ONNX output".into());
        }
        for row in 0..query.len() {
            if task == Task::Classification {
                scores.push((output[(row, 1)] - output[(row, 0)]) as f64);
            } else {
                let mut quantiles = output.row(row).iter().copied().collect::<Vec<_>>();
                quantiles.sort_by(f32::total_cmp);
                predictions.push(quantiles[499] as f64 * scale + center);
            }
        }
    }
    Ok(Inference {
        predictions,
        scores,
        seconds: start.elapsed().as_secs_f64(),
    })
}

fn classification_counts(truth: &[f64], scores: &[f64], threshold: f64) -> [usize; 4] {
    let (mut tn, mut fp, mut fn_, mut tp) = (0, 0, 0, 0);
    for (&actual, &score) in truth.iter().zip(scores) {
        match (actual == 1.0, score > threshold) {
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (true, true) => tp += 1,
        }
    }
    [tn, fp, fn_, tp]
}

fn accuracy(truth: &[f64], scores: &[f64], threshold: f64) -> f64 {
    let [tn, _, _, tp] = classification_counts(truth, scores, threshold);
    (tn + tp) as f64 / truth.len() as f64
}

fn balanced_accuracy(truth: &[f64], scores: &[f64], threshold: f64) -> f64 {
    let [tn, fp, fn_, tp] = classification_counts(truth, scores, threshold);
    (tn as f64 / (tn + fp) as f64 + tp as f64 / (tp + fn_) as f64) / 2.0
}

fn select_threshold(truth: &[f64], scores: &[f64]) -> f64 {
    let mut candidates = scores.to_vec();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    // Balance sensitivity and specificity so the majority class cannot
    // trivially win by making every prediction negative.
    let mut best = *candidates.last().expect("validation set must be nonempty");
    let mut best_accuracy = balanced_accuracy(truth, scores, best);
    for &candidate in &candidates {
        let value = balanced_accuracy(truth, scores, candidate);
        if value > best_accuracy {
            best = candidate;
            best_accuracy = value;
        }
    }
    best
}

fn mean_absolute_error(truth: &[f64], predictions: &[f64]) -> f64 {
    truth
        .iter()
        .zip(predictions)
        .map(|(y, p)| (y - p).abs())
        .sum::<f64>()
        / truth.len() as f64
}

fn validation_split(
    train: &[usize],
    targets: &[f64],
    classification: bool,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A_5A5A_A5A5);
    let mut fit = Vec::new();
    let mut validation = Vec::new();
    for label in if classification {
        vec![Some(0.0), Some(1.0)]
    } else {
        vec![None]
    } {
        let mut indices = train
            .iter()
            .copied()
            .filter(|&index| label.is_none_or(|value| targets[index] == value))
            .collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        let count = (indices.len() as f64 * 0.2).round() as usize;
        validation.extend_from_slice(&indices[..count]);
        fit.extend_from_slice(&indices[count..]);
    }
    validation.shuffle(&mut rng);
    (fit, validation)
}

fn evaluate(task: Task, size: ModelSize, seed: u64) -> EvalResult<Value> {
    let classification = task == Task::Classification;
    let (path, features, target) = if classification {
        (
            "dataset/classification/bad_customer_detection.parquet",
            ["month", "credit_amount", "age"],
            "bad_customer",
        )
    } else {
        (
            "dataset/regression/airfoil_self_noise.parquet",
            [
                "frequency",
                "attack-angle",
                "suction-side-displacement-thickness",
            ],
            "scaled-sound-pressure",
        )
    };
    let model = size.model_path(task);
    let data = load_data(path, features, target, classification)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let (train, test) = split(&data, classification, &mut rng);
    if train.len() < 4 || test.is_empty() {
        return Err("insufficient train/test rows".into());
    }
    let (fit, validation) = validation_split(&train, &data.targets, classification, seed);
    let x = standardize(&data, &fit);
    let mut session = OnnxSession::load(model)?;
    if session.input_names() != ["x", "y", "categorical_mask"]
        || session.output_names() != ["predictions"]
    {
        return Err("unexpected model tensor contract".into());
    }
    if classification
        && [0.0, 1.0]
            .iter()
            .any(|&label| fit.iter().filter(|&&i| data.targets[i] == label).count() < 2)
    {
        return Err("classification requires two training samples per class".into());
    }
    let validation_truth = validation
        .iter()
        .map(|&i| data.targets[i])
        .collect::<Vec<_>>();
    let mut selected = None;
    for strategy in [ContextSelection::Random, ContextSelection::Nearest] {
        let inferred = infer(
            &mut session,
            &data,
            &x,
            &fit,
            &validation,
            task,
            strategy,
            seed,
        )?;
        let threshold = if classification {
            select_threshold(&validation_truth, &inferred.scores)
        } else {
            0.0
        };
        let loss = if classification {
            1.0 - auc(&validation_truth, &inferred.scores)
        } else {
            mean_absolute_error(&validation_truth, &inferred.predictions)
        };
        if selected
            .as_ref()
            .is_none_or(|(_, _, best_loss): &(ContextSelection, f64, f64)| loss < *best_loss)
        {
            selected = Some((strategy, threshold, loss));
        }
    }
    let (strategy, threshold, validation_loss) = selected.ok_or("no context strategy evaluated")?;
    let inferred = infer(&mut session, &data, &x, &fit, &test, task, strategy, seed)?;
    let predictions = inferred.predictions;
    let scores = inferred.scores;
    let truth = test.iter().map(|&i| data.targets[i]).collect::<Vec<_>>();
    let metrics = if classification {
        let [tn, fp, fn_, tp] = classification_counts(&truth, &scores, threshold);
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = tp as f64 / (tp + fn_) as f64;
        json!({
            "accuracy": accuracy(&truth, &scores, threshold),
            "balanced_accuracy": (recall + tn as f64 / (tn + fp) as f64) / 2.0,
            "precision_yes": precision, "recall_yes": recall,
            "f1_yes": if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) },
            "roc_auc_yes": auc(&truth, &scores),
            "confusion_matrix_no_yes": [[tn, fp], [fn_, tp]],
            "majority_no_accuracy": (tn + fp) as f64 / test.len() as f64,
        })
    } else {
        let n = test.len() as f64;
        let mean = truth.iter().sum::<f64>() / n;
        let train_mean = train.iter().map(|&i| data.targets[i]).sum::<f64>() / train.len() as f64;
        let squared_error = predictions
            .iter()
            .zip(&truth)
            .map(|(p, y)| (p - y).powi(2))
            .sum::<f64>();
        json!({
            "mae": predictions.iter().zip(&truth).map(|(p, y)| (p - y).abs()).sum::<f64>() / n,
            "rmse": (squared_error / n).sqrt(),
            "r2": 1.0 - squared_error / truth.iter().map(|y| (y - mean).powi(2)).sum::<f64>(),
            "train_mean_baseline_mae": truth.iter().map(|y| (y - train_mean).abs()).sum::<f64>() / n,
            "train_mean_baseline_rmse": (truth.iter().map(|y| (y - train_mean).powi(2)).sum::<f64>() / n).sqrt(),
        })
    };
    Ok(json!({
        "task": if classification { "classification" } else { "regression" },
        "engine": "rust-ort", "size": size.as_str(), "precision": "fp16", "dataset": path, "model": model,
        "features": features, "target": target, "seed": seed,
        "train_rows": train.len(), "fit_rows": fit.len(), "validation_rows": validation.len(), "test_rows": test.len(),
        "context_strategy": strategy.as_str(),
        "validation_objective": if classification { "roc_auc" } else { "mae" },
        "threshold_objective": if classification { Some("balanced_accuracy") } else { None },
        "validation_loss": validation_loss,
        "classification_threshold": if classification { Some(threshold) } else { None },
        "context_rows_per_batch": 4, "queries_per_batch": 2,
        "prediction_seconds": (inferred.seconds * 1000.0).round() / 1000.0,
        "metrics": metrics,
    }))
}

fn main() -> EvalResult<()> {
    let mut args = std::env::args().skip(1);
    let mut seed = 42;
    let mut size = "both".to_string();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => seed = args.next().ok_or("missing --seed value")?.parse()?,
            "--size" => size = args.next().ok_or("missing --size value")?,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let sizes = if size == "both" {
        vec![ModelSize::Small, ModelSize::Medium]
    } else {
        vec![ModelSize::parse(&size)?]
    };
    for size in sizes {
        let mut results = Vec::new();
        for task in [Task::Classification, Task::Regression] {
            results.push(evaluate(task, size, seed)?);
        }
        let output = format!("output/{}-evaluation-rust.json", size.as_str());
        let text = serde_json::to_string_pretty(&results)?;
        if let Some(parent) = Path::new(&output).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, format!("{text}\n"))?;
        println!("{} report: {output}\n{text}", size.as_str());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_handles_tied_scores() {
        assert!((auc(&[0.0, 1.0, 0.0, 1.0], &[0.0, 0.0, 1.0, 1.0]) - 0.5).abs() < 1e-12);
        assert!((auc(&[0.0, 0.0, 1.0, 1.0], &[0.0, 0.1, 0.8, 1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn split_never_leaks_test_rows_into_training() {
        let data = Dataset {
            features: vec![[0.0; 3]; 20],
            targets: (0..20).map(|i| (i % 2) as f64).collect(),
        };
        let (train, test) = split(&data, true, &mut StdRng::seed_from_u64(42));
        assert_eq!((train.len(), test.len()), (16, 4));
        assert!(train.iter().all(|i| !test.contains(i)));
        assert_eq!(test.iter().filter(|&&i| data.targets[i] == 1.0).count(), 2);
    }

    #[test]
    fn validation_rows_are_disjoint_from_fit_and_test() {
        let data = Dataset {
            features: vec![[0.0; 3]; 100],
            targets: (0..100).map(|i| (i % 10 == 0) as u8 as f64).collect(),
        };
        let (train, test) = split(&data, true, &mut StdRng::seed_from_u64(42));
        let (fit, validation) = validation_split(&train, &data.targets, true, 42);
        assert_eq!(fit.len() + validation.len(), train.len());
        assert!(fit
            .iter()
            .all(|i| !validation.contains(i) && !test.contains(i)));
        assert!(validation.iter().all(|i| !test.contains(i)));
    }

    #[test]
    fn balanced_threshold_keeps_positive_predictions() {
        let truth = [0.0, 0.0, 0.0, 0.0, 1.0];
        let scores = [0.1, 0.2, 0.3, 0.4, 0.25];
        let threshold = select_threshold(&truth, &scores);
        assert!(threshold < 0.25);
        assert!(balanced_accuracy(&truth, &scores, threshold) > 0.5);
        assert!(accuracy(&truth, &scores, threshold) < 0.8);
    }

    #[test]
    fn nearest_context_uses_only_training_rows() {
        let features = vec![
            [0.0; 3], [0.2; 3], [0.3; 3], [0.4; 3], [8.0; 3], [9.0; 3], [0.1; 3],
        ];
        let targets = vec![0.0; features.len()];
        let context = choose_context(
            &[0, 1, 2, 3, 4, 5],
            &[6],
            &features,
            &targets,
            Task::Regression,
            ContextSelection::Nearest,
            &mut StdRng::seed_from_u64(42),
        );
        assert_eq!(context.len(), 4);
        assert!(context.iter().all(|&index| index < 4));
        assert!(!context.contains(&6));
    }

    #[test]
    fn model_paths_cover_both_tasks_and_sizes() {
        for task in [Task::Classification, Task::Regression] {
            for size in [ModelSize::Small, ModelSize::Medium] {
                assert!(size.model_path(task).ends_with("-fp16.onnx"));
                assert!(size.model_path(task).contains(size.as_str()));
                assert!(size.model_path(task).starts_with(size.as_str()));
            }
        }
    }

    #[test]
    #[ignore = "requires local small/medium FP16 ONNX models and ORT_DYLIB_PATH; run with --ignored"]
    fn quantized_models_evaluate_both_sizes_and_datasets() {
        assert!(
            std::env::var_os("ORT_DYLIB_PATH").is_some(),
            "set ORT_DYLIB_PATH before running this test"
        );
        for size in [ModelSize::Small, ModelSize::Medium] {
            for task in [Task::Classification, Task::Regression] {
                assert!(
                    Path::new(size.model_path(task)).exists(),
                    "missing model: {}",
                    size.model_path(task)
                );
                let result = evaluate(task, size, 42).expect("FP16 inference should succeed");
                assert_eq!(result["size"], size.as_str());
                assert_eq!(result["precision"], "fp16");
                assert!(result["test_rows"].as_u64().unwrap() > 0);
                assert!(result["metrics"].is_object());
            }
        }
    }
}
