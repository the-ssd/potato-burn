use std::collections::HashMap;

use burn::train::{
    metric::{MetricDefinition, MetricId},
    renderer::{
        EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
        MetricsRendererEvaluation, MetricsRendererTraining, ProgressType, TrainingProgress,
    },
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer {
    items_train: HashMap<MetricId, String>,
    items_valid: HashMap<MetricId, String>,
    id_to_string: HashMap<MetricId, String>,
}

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {
            items_train: HashMap::new(),
            items_valid: HashMap::new(),
            id_to_string: HashMap::new(),
        }
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        //dbg!(state);
        match state {
            MetricState::Generic(metric_entry) => self.items_train.insert(
                metric_entry.metric_id,
                metric_entry.serialized_entry.formatted,
            ),
            MetricState::Numeric(metric_entry, _numeric_entry) => self.items_train.insert(
                metric_entry.metric_id,
                metric_entry.serialized_entry.formatted,
            ),
        };
    }

    fn update_valid(&mut self, state: MetricState) {
        //dbg!(state);
        match state {
            MetricState::Generic(metric_entry) => self.items_valid.insert(
                metric_entry.metric_id,
                metric_entry.serialized_entry.formatted,
            ),
            MetricState::Numeric(metric_entry, _numeric_entry) => self.items_valid.insert(
                metric_entry.metric_id,
                metric_entry.serialized_entry.formatted,
            ),
        };
    }

    fn render_train(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        println!();
        for (item_id, item) in &self.items_train {
            println!("{}: {}", self.id_to_string[item_id], item);
        }
        if let Some(progress) = item.progress {
            println!(
                "Epoch: {}/{}",
                progress.items_processed, progress.items_total
            );
        }
        println!(
            "Training run: {}/{}",
            item.global_progress.items_processed, item.global_progress.items_total
        );
    }

    fn render_valid(&mut self, item: TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        println!();
        for (item_id, item) in &self.items_valid {
            println!("{}: {}", self.id_to_string[item_id], item);
        }
        if let Some(progress) = item.progress {
            println!(
                "Epoch: {}/{}",
                progress.items_processed, progress.items_processed
            );
        }
        println!(
            "Training run: {}/{}",
            item.global_progress.items_processed, item.global_progress.items_processed
        );
    }
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn render_test(&mut self, item: EvaluationProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
        dbg!(&self.items_valid);
        dbg!(&self.items_train);
    }

    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {
        dbg!(_state);
    }
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }

    fn register_metric(&mut self, definition: MetricDefinition) {
        self.id_to_string
            .insert(definition.metric_id, definition.name);
        //println!("Registered {}", definition.name);
    }
}
