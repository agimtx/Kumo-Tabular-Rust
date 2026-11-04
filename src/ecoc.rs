//! Error-correcting output codes for classification targets above the ten
//! classes supported directly by the exported Kumo core.

use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};

/// Number of output channels in the exported Kumo classifier.
pub const CORE_CLASSES: usize = 10;
/// Symbol reserved for "the rest of the classes" inside one ECOC task.
pub const REST_SYMBOL: u8 = CORE_CLASSES as u8 - 1;

/// A deterministic ECOC codebook: one symbol in `0..CORE_CLASSES` per class and
/// task. Every task activates nine individually scored classes and puts the
/// remaining classes into the rest bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codebook {
    tasks: Vec<Vec<u8>>,
    num_classes: usize,
}

impl Codebook {
    /// Build a codebook for `num_classes` classes. The task count follows the
    /// upstream Kumo/TabPFN rule: enough tasks to give every class its own output
    /// at least once, with a logarithmic lower bound for large target spaces.
    pub fn build(num_classes: usize, seed: u64) -> Self {
        assert!(
            num_classes > CORE_CLASSES,
            "ECOC is only needed above {CORE_CLASSES} classes"
        );
        let active_per_task = CORE_CLASSES - 1;
        let mut rng = StdRng::seed_from_u64(seed);
        let log_tasks =
            4 * ((num_classes as f64).ln() / (CORE_CLASSES as f64).ln()).ceil() as usize;
        let target_tasks = std::cmp::max(num_classes.div_ceil(active_per_task), log_tasks);

        let mut coverage = vec![0usize; num_classes];
        let mut tasks: Vec<Vec<u8>> = Vec::with_capacity(target_tasks);
        while tasks.len() < target_tasks || coverage.contains(&0) {
            // Safety valve for pathological cases; the loop below normally
            // converges after one or two rounds.
            if tasks.len() > num_classes + target_tasks + 8 {
                break;
            }
            let mut candidates = (0..num_classes)
                .map(|class| (coverage[class] as f64 + 0.1 * rng.gen::<f64>(), class))
                .collect::<Vec<_>>();
            candidates.sort_by(
                |(left_priority, left_class), (right_priority, right_class)| {
                    left_priority
                        .total_cmp(right_priority)
                        .then_with(|| left_class.cmp(right_class))
                },
            );

            let mut symbols = (0..active_per_task as u8).collect::<Vec<_>>();
            symbols.shuffle(&mut rng);
            let mut task = vec![REST_SYMBOL; num_classes];
            for (symbol, &(_, class)) in symbols.iter().zip(candidates.iter()) {
                task[class] = *symbol;
                coverage[class] += 1;
            }
            tasks.push(task);
        }

        debug_assert!(coverage.iter().all(|&count| count > 0));
        Self { tasks, num_classes }
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn symbol(&self, task: usize, class: usize) -> u8 {
        self.tasks[task][class]
    }

    pub fn task(&self, task: usize) -> &[u8] {
        &self.tasks[task]
    }
}

/// Stable log-softmax for one row of core logits.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer = logits
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f32>()
        .ln();
    logits
        .iter()
        .map(|value| value - max - normalizer)
        .collect()
}

/// Aggregate per-task core logits into class probabilities.
///
/// `logits_by_task[t]` is row-major `[rows, 10]` for task `t`. Each class
/// accumulates the log-probability of its active symbol in every task where it
/// is not the rest bucket; the mean log-probability is then softmaxed.
pub fn aggregate(codebook: &Codebook, logits_by_task: &[Vec<f32>], rows: usize) -> Vec<f32> {
    let classes = codebook.num_classes();
    let mut sums = vec![0.0f64; rows * classes];
    let mut counts = vec![0usize; rows * classes];

    for (task_index, logits) in logits_by_task.iter().enumerate() {
        debug_assert_eq!(logits.len(), rows * CORE_CLASSES);
        for row in 0..rows {
            let row_logits = &logits[row * CORE_CLASSES..(row + 1) * CORE_CLASSES];
            let log_probs = log_softmax(row_logits);
            for class in 0..classes {
                let symbol = codebook.symbol(task_index, class) as usize;
                if symbol != REST_SYMBOL as usize {
                    sums[row * classes + class] += log_probs[symbol] as f64;
                    counts[row * classes + class] += 1;
                }
            }
        }
    }

    let mut probabilities = vec![0.0f32; rows * classes];
    for row in 0..rows {
        let mut max = f64::NEG_INFINITY;
        for class in 0..classes {
            let index = row * classes + class;
            let score = if counts[index] == 0 {
                f64::NEG_INFINITY
            } else {
                sums[index] / counts[index] as f64
            };
            sums[index] = score;
            if score > max {
                max = score;
            }
        }
        let mut total = 0.0f64;
        for class in 0..classes {
            let index = row * classes + class;
            let value = (sums[index] - max).exp();
            sums[index] = value;
            total += value;
        }
        for class in 0..classes {
            let index = row * classes + class;
            probabilities[index] = if total > 0.0 {
                (sums[index] / total) as f32
            } else {
                0.0
            };
        }
    }
    probabilities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_class_is_active_somewhere() {
        for classes in [11usize, 12, 19, 37, 100, 257] {
            let codebook = Codebook::build(classes, 7);
            assert!(codebook.task_count() >= classes.div_ceil(9));
            for class in 0..classes {
                let active = (0..codebook.task_count())
                    .filter(|&task| codebook.symbol(task, class) != REST_SYMBOL)
                    .count();
                assert!(active > 0, "class {class} is never active");
            }
        }
    }

    #[test]
    fn each_task_has_nine_active_symbols_and_one_rest_bucket() {
        let codebook = Codebook::build(23, 1);
        for task in 0..codebook.task_count() {
            let symbols = codebook.task(task);
            let mut seen = [false; CORE_CLASSES];
            for &symbol in symbols {
                assert!((symbol as usize) < CORE_CLASSES);
                seen[symbol as usize] = true;
            }
            // Nine symbols are assigned to classes; the unused symbol is the
            // rest bucket. With 23 classes every symbol 0..8 must occur.
            for symbol in 0..REST_SYMBOL {
                assert!(seen[symbol as usize], "task {task} lacks symbol {symbol}");
            }
        }
    }

    #[test]
    fn aggregation_returns_normalized_probabilities() {
        let classes = 11;
        let codebook = Codebook::build(classes, 3);
        let tasks = codebook.task_count();
        let logits = (0..tasks)
            .map(|task| {
                (0..CORE_CLASSES)
                    .map(|symbol| (task as f32 * 0.3) + (symbol as f32 * 0.1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let probabilities = aggregate(&codebook, &logits, 1);
        assert_eq!(probabilities.len(), classes);
        let total = probabilities.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-5, "probabilities sum to {total}");
        assert!(probabilities.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn log_softmax_matches_softmax_log() {
        let logits = [0.5f32, -1.0, 2.0, 0.0];
        let log_probs = log_softmax(&logits);
        let total = logits.iter().map(|v| v.exp()).sum::<f32>();
        for (value, log_prob) in logits.iter().zip(&log_probs) {
            assert!((log_prob - (value - total.ln())).abs() < 1e-4);
        }
    }
}
