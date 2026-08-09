use burn::train::{
    metric::MetricDefinition,
    renderer::{
        EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
        MetricsRendererEvaluation, MetricsRendererTraining, ProgressType, TrainingProgress,
    },
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer {}

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        dbg!(state);
        /*match state {
            MetricState::Generic(metric_entry) => todo!(),
            MetricState::Numeric(metric_entry, numeric_entry) => todo!(),
        }*/
    }

    fn update_valid(&mut self, state: MetricState) {
        dbg!(state);
        /*match state {
            MetricState::Generic(metric_entry) => todo!(),
            MetricState::Numeric(metric_entry, numeric_entry) => todo!(),
        }*/
    }

    fn render_train(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }

    fn render_valid(&mut self, item: TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn render_test(&mut self, item: EvaluationProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }

    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {
        dbg!(_state);
    }
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }

    fn register_metric(&mut self, _definition: MetricDefinition) {
        println!("Register");
    }
}
