use csv::ReaderBuilder;
use hyper::body::Bytes;
use linfa::dataset::Dataset;
use linfa::metrics::SingleTargetRegression;
use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa_linear::{FittedLinearRegression, LinearRegression};
use ndarray::{Array1, Array2, Ix1};
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::{create_dir, create_dir_all};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    shape: (usize, usize),
    target: String,
    target_type: String,
    unique_values: Vec<String>,
    train_size: usize,
    test_size: usize,
    validation_size: usize,
    l2_regularization: f32,
    learning_rate: f32,
    max_epochs: usize,
    batch_size: usize,
    early_stopping: Option<EarlyStopping>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, PartialOrd)]
pub enum Metrics {
    LinearRegression {
        params: Vec<f32>,
        intercept: f32,
        max_error: f32,
        mean_absolute_error: f32,
        mean_squared_error: f32,
        mean_squared_log_error: f32,
        median_absolute_error: f32,
        mean_absolute_percentage_error: f32,
        r2_score: f32,
        explained_variance_score: f32,
    },
    LogisticRegression {
        params: Vec<f32>,
        intercept: f32,
        train_accuracy: f32,
        test_accuracy: f32,
    },
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::LinearRegression {
            params: vec![],
            intercept: 0.0,
            max_error: 0.0,
            mean_absolute_error: 0.0,
            mean_squared_error: 0.0,
            mean_squared_log_error: 0.0,
            median_absolute_error: 0.0,
            mean_absolute_percentage_error: 0.0,
            r2_score: 0.0,
            explained_variance_score: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum TrainedModel {
    LinearRegression(FittedLinearRegression<f32>),
}

impl TrainedModel {
    pub fn predict(&self, data: &Array2<f32>) -> Array1<f32> {
        match self {
            TrainedModel::LinearRegression(model) => model.predict(data),
        }
    }

    pub fn load(path: &PathBuf) -> Self {
        let file = File::open(path).unwrap();
        let model: TrainedModel = bincode::deserialize_from(file).unwrap();
        model
    }

    pub fn save(&self, path: &PathBuf) {
        let _encoded: Vec<u8> = bincode::serialize(&self).unwrap();
        let mut file = File::create(path.clone()).unwrap();
        bincode::serialize_into(&mut file, &_encoded).unwrap();
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ModelType {
    LinearRegression,
    LogisticRegression,
    RandomForest,
    DecisionTree,
    SVM,
    KNN,
    KMeans,
    PCA,
    TSNE,
    UMAP,
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::LinearRegression
    }
}

impl ToString for ModelType {
    fn to_string(&self) -> String {
        match self {
            ModelType::LinearRegression => "Linear Regression".to_string(),
            ModelType::LogisticRegression => "Logistic Regression".to_string(),
            ModelType::RandomForest => "Random Forest".to_string(),
            ModelType::DecisionTree => "Decision Tree".to_string(),
            ModelType::SVM => "SVM".to_string(),
            ModelType::KNN => "KNN".to_string(),
            ModelType::KMeans => "KMeans".to_string(),
            ModelType::PCA => "PCA".to_string(),
            ModelType::TSNE => "TSNE".to_string(),
            ModelType::UMAP => "UMAP".to_string(),
        }
    }
}

impl ModelType {
    pub fn train(&self, dataset: &Dataset<f32, f32, Ix1>) -> Option<TrainedModel> {
        match self {
            ModelType::LinearRegression => {
                let model = LinearRegression::default().fit(&dataset).unwrap();
                Some(TrainedModel::LinearRegression(model))
            }
            _ => {
                tracing::error!("Model type not implemented");
                None
            }
        }
    }

    pub fn evaluate(&self, model: &TrainedModel, dataset: &Dataset<f32, f32, Ix1>) -> Metrics {
        match self {
            ModelType::LinearRegression => {
                let model = match model {
                    TrainedModel::LinearRegression(model) => model,
                };
                let predictions = model.predict(&dataset.records);
                Metrics::LinearRegression {
                    params: model.params().to_vec(),
                    intercept: model.intercept(),
                    max_error: dataset.max_error(&predictions).unwrap(),
                    mean_absolute_error: dataset.mean_absolute_error(&predictions).unwrap(),
                    mean_squared_error: dataset.mean_squared_error(&predictions).unwrap(),
                    mean_squared_log_error: dataset.mean_squared_log_error(&predictions).unwrap(),
                    median_absolute_error: dataset.median_absolute_error(&predictions).unwrap(),
                    mean_absolute_percentage_error: dataset
                        .mean_absolute_percentage_error(&predictions)
                        .unwrap(),
                    r2_score: dataset.r2(&predictions).unwrap(),
                    explained_variance_score: dataset.explained_variance(&predictions).unwrap(),
                }
            }
            _ => {
                tracing::error!("Model type not implemented");
                Metrics::default()
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, Eq)]
pub struct Model {
    id: Uuid,
    name: String,
    model_type: ModelType,
    data_hash: String,
    path: PathBuf,
    config: Value,
    metrics: Value,
}

impl PartialEq for Model {
    fn eq(&self, other: &Model) -> bool {
        self.id == other.id
    }
}

impl Hash for Model {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Borrow<Uuid> for Model {
    fn borrow(&self) -> &Uuid {
        &self.id
    }
}

impl Model {
    pub async fn new(
        state: Arc<Mutex<AppState>>,
        project: Uuid,
        data_hash: &str,
        config: ModelConfig,
    ) -> Self {
        let state = state.lock().await;
        let id = Uuid::new_v4();
        create_dir(state.get_path().join(project.to_string()))
            .await
            .unwrap();
        let path = state
            .get_path()
            .join(project.to_string())
            .join(format!("{}.bin", id.simple().to_string()));
        let name = config.name.clone();
        let model_type = config.model_type.clone();
        let data_hash = data_hash.to_string();

        let datahandle = state
            .get_project(&project)
            .unwrap()
            .get_datahandle(&data_hash)
            .unwrap();

        let (data, targets) = datahandle.get_data();
        let feature_names = datahandle.get_features();
        let dataset = Dataset::new(data, targets).with_feature_names(feature_names);
        drop(state);

        let model = model_type.train(&dataset).unwrap();
        let metrics = model_type.evaluate(&model, &dataset);
        model.save(&path);
        Self {
            id,
            name,
            model_type,
            data_hash,
            path,
            config: json!(config),
            metrics: json!(metrics),
        }
    }

    pub fn get_id(&self) -> Uuid {
        self.id
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    pub fn get_data_hash(&self) -> String {
        self.data_hash.clone()
    }

    pub fn get_path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn get_config(&self) -> ModelConfig {
        serde_json::from_value(self.config.clone()).unwrap()
    }

    pub fn get_metrics(&self) -> Metrics {
        serde_json::from_value(self.metrics.clone()).unwrap()
    }

    pub fn get_config_json(&self) -> Value {
        self.config.clone()
    }

    pub fn get_metrics_json(&self) -> Value {
        self.metrics.clone()
    }

    pub fn get_info(&self) -> Value {
        json!({
            "id": self.id.to_string(),
            "name": self.name,
            "model_type": self.model_type.to_string(),
            "data_hash": self.data_hash,
        })
    }

    pub async fn get_overview(&self) -> Value {
        json!({
            "name": self.name,
            "model_type": self.model_type.to_string(),
            "data_hash": self.data_hash,
            "config": self.config,
            "metrics": self.metrics,
        })
    }

    pub fn predict(&self, data: &Array2<f32>) -> Array1<f32> {
        match self.model_type {
            ModelType::LinearRegression => {
                let model = TrainedModel::load(&self.path);
                model.predict(data)
            }
            _ => {
                tracing::error!("Model type not implemented");
                Array1::zeros(data.nrows())
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, Default)]
pub struct DataHandle {
    path: PathBuf,
    data: PathBuf,
    targets: PathBuf,
    content_type: String,
    hash: String,
    shape: (usize, usize),
    features: Vec<String>,
    target: String,
    head: Vec<Vec<String>>,
}

impl PartialEq for DataHandle {
    fn eq(&self, other: &DataHandle) -> bool {
        self.hash == other.hash
    }
}

impl Hash for DataHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl Borrow<str> for DataHandle {
    fn borrow(&self) -> &str {
        &self.hash
    }
}

impl DataHandle {
    pub async fn new(
        path: PathBuf,
        name: String,
        content_type: String,
        data: Bytes,
        target: String,
    ) -> Result<Self, String> {
        let mut data_handle = Self::default();
        let file_path = path.join(name);
        if file_path.exists() {
            return Err("File already exists".to_string());
        }
        create_dir_all(&path).await.unwrap();
        tokio::fs::write(&file_path, data).await.unwrap();
        let hash = sha256::try_async_openssl_digest(&file_path).await.unwrap();
        data_handle.path = file_path;
        data_handle.content_type = content_type.to_string();
        data_handle.hash = hash;

        match content_type.as_str() {
            "text/csv" => {
                let mut csv_file = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(&data_handle.path)
                    .unwrap();
                let header = csv_file.headers().unwrap().clone();
                let records = csv_file
                    .into_records()
                    .filter_map(|r| r.ok())
                    .collect::<Vec<_>>();
                let records = records
                    .into_iter()
                    .map(|record| record.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                    .collect::<Vec<_>>();

                data_handle.shape = (records.len(), header.len());
                data_handle.features = header
                    .iter()
                    .map(|s| s.to_string())
                    .filter(|s| s != &target)
                    .collect();
                data_handle.target = target.to_string();
                data_handle.head = records.clone().into_iter().take(5).collect();

                let array: Array2<f32> = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(&data_handle.path)
                    .unwrap()
                    .deserialize_array2(data_handle.shape)
                    .unwrap();

                let targets = array
                    .column(header.iter().position(|s| s == &target).unwrap())
                    .to_vec()
                    .into_iter()
                    .collect::<Vec<_>>();
                let data = array
                    .columns()
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| *i != header.iter().position(|s| s == &target).unwrap())
                    .map(|(_, c)| c.to_vec().into_iter().collect::<Vec<_>>())
                    .collect::<Vec<_>>();

                data_handle.data = data_handle.path.clone();
                data_handle.data.set_extension("data.bin");
                data_handle.targets = data_handle.path.clone();
                data_handle.targets.set_extension("targets.bin");

                let mut data_file = File::create(&data_handle.data).unwrap();
                let mut targets_file = File::create(&data_handle.targets).unwrap();

                bincode::serialize_into(&mut data_file, &data).unwrap();
                bincode::serialize_into(&mut targets_file, &targets).unwrap();
                Ok(data_handle)
            }
            _ => Err("Unsupported content type".to_string()),
        }
    }

    pub fn get_path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn get_name(&self) -> String {
        self.path.file_name().unwrap().to_str().unwrap().to_string()
    }

    pub fn get_size(&self) -> u64 {
        self.path.metadata().unwrap().len()
    }

    pub fn get_content_type(&self) -> String {
        self.content_type.clone()
    }

    pub fn get_hash(&self) -> String {
        self.hash.clone()
    }

    pub fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn get_features(&self) -> Vec<String> {
        self.features.clone()
    }

    pub fn get_target(&self) -> String {
        self.target.clone()
    }

    pub fn get_head(&self) -> Vec<Vec<String>> {
        self.head.clone()
    }

    pub fn get_info(&self) -> Value {
        json!({
            "name": self.get_name(),
            "size": self.get_size(),
            "content_type": self.get_content_type(),
            "hash": self.get_hash(),
            "shape": self.get_shape(),
            "features": self.get_features(),
            "target": self.get_target(),
            "head": self.get_head(),
        })
    }

    pub fn get_data(&self) -> (Array2<f32>, Array1<f32>) {
        let data_file = File::open(&self.data).unwrap();
        let targets_file = File::open(&self.targets).unwrap();

        let data: Vec<Vec<f32>> = bincode::deserialize_from(data_file).unwrap();
        let data: Vec<f32> = data.into_iter().flatten().collect();
        let targets: Vec<f32> = bincode::deserialize_from(targets_file).unwrap();

        let data = Array2::from_shape_vec((self.shape.0, self.shape.1 - 1), data).unwrap();
        let targets = Array1::from(targets);

        (data, targets)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, Eq)]
pub struct Project {
    id: Uuid,
    name: String,
    description: String,
    expires: i64,
    data: HashSet<DataHandle>,
    models: HashSet<Model>,
}

impl PartialEq for Project {
    fn eq(&self, other: &Project) -> bool {
        self.id == other.id
    }
}

impl Hash for Project {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Borrow<Uuid> for Project {
    fn borrow(&self) -> &Uuid {
        &self.id
    }
}

impl Project {
    pub fn new(name: String, description: String, expires: i64) -> Self {
        Project {
            id: Uuid::new_v4(),
            name,
            description,
            expires,
            ..Default::default()
        }
    }

    pub fn get_id(&self) -> Uuid {
        self.id
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_description(&self) -> String {
        self.description.clone()
    }

    pub fn get_expires(&self) -> i64 {
        self.expires
    }

    pub fn get_datahandles(&self) -> Vec<&DataHandle> {
        self.data.iter().collect()
    }

    pub fn get_datahandle(&self, hash: &str) -> Option<&DataHandle> {
        self.data.get(hash)
    }

    pub fn get_info(&self) -> Value {
        json!({
            "id": self.id.to_string(),
            "name": self.name,
            "description": self.description,
            "expires": self.expires,
        })
    }

    pub fn get_models(&self, hash: &str) -> Vec<&Model> {
        self.models
            .iter()
            .filter(|m| m.get_data_hash() == hash)
            .collect()
    }

    pub fn get_model(&self, id: Uuid) -> Option<&Model> {
        self.models.get(&id)
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn set_description(&mut self, description: String) {
        self.description = description;
    }

    fn add_datahandle(&mut self, data: DataHandle) {
        self.data.insert(data);
    }

    fn remove_datahandle(&mut self, hash: &str) -> Option<DataHandle> {
        self.data.take(hash)
    }

    fn add_model(&mut self, model: Model) {
        self.models.insert(model);
    }

    fn remove_model(&mut self, id: Uuid) -> Option<Model> {
        self.models.take(&id)
    }
}

#[derive(Deserialize, Serialize, Clone, Default, Debug)]
pub struct AppState {
    server_path: PathBuf,
    projects: HashSet<Project>,
    #[serde(skip)]
    tx: Option<tokio::sync::mpsc::Sender<(Uuid, String, ModelConfig)>>,
}

impl AppState {
    pub async fn new(
        server_path: PathBuf,
        tx: tokio::sync::mpsc::Sender<(Uuid, String, ModelConfig)>,
    ) -> Self {
        let mut state = AppState {
            server_path,
            tx: Some(tx),
            ..Default::default()
        };

        if state.server_path.join("config.json").exists() {
            state.load().await.unwrap();
        } else {
            state.write().await.unwrap();
        }

        state
    }

    async fn write(&self) -> tokio::io::Result<()> {
        tokio::fs::write(
            self.server_path.join("config.json"),
            serde_json::to_string(&self).unwrap(),
        )
        .await
    }

    async fn load(&mut self) -> tokio::io::Result<()> {
        let config = tokio::fs::read_to_string(self.server_path.join("config.json")).await?;
        let config: AppState = serde_json::from_str(&config).unwrap();
        self.server_path = config.server_path;
        self.projects = config.projects;
        Ok(())
    }

    pub fn get_path(&self) -> PathBuf {
        self.server_path.clone()
    }

    pub fn get_projects(&self) -> Vec<&Project> {
        self.projects.iter().collect()
    }

    pub fn get_project(&self, id: &Uuid) -> Option<&Project> {
        self.projects.get(id)
    }

    pub async fn add(&mut self, project: Project) -> Uuid {
        let id = Uuid::new_v4();
        let project = Project { id, ..project };
        self.projects.insert(project);
        create_dir_all(self.server_path.join(&id.to_string()))
            .await
            .unwrap();
        self.write().await.unwrap();
        id
    }

    pub async fn remove(&mut self, id: Uuid) -> Option<Project> {
        let project = self.projects.take(&id);
        match project {
            Some(project) => {
                tokio::fs::remove_dir_all(self.server_path.join(&id.to_string()))
                    .await
                    .unwrap();
                self.write().await.unwrap();
                Some(project)
            }
            None => None,
        }
    }

    pub async fn update(&mut self, project: Project) {
        self.projects.insert(project);
        self.write().await.unwrap();
    }

    pub async fn add_datahandle(&mut self, id: Uuid, data: DataHandle) {
        let mut project = self.projects.take(&id).unwrap();
        project.add_datahandle(data);
        self.projects.insert(project);
        self.write().await.unwrap();
    }

    pub async fn remove_datahandle(&mut self, id: Uuid, hash: &str) {
        let mut project = self.projects.take(&id).unwrap();
        let data_handle = project.remove_datahandle(hash);
        match data_handle {
            Some(data_handle) => {
                tokio::fs::remove_file(data_handle.get_path())
                    .await
                    .unwrap();
            }
            None => {}
        }
        self.projects.insert(project);
        self.write().await.unwrap();
    }

    pub async fn add_model_queue(&self, id: Uuid, hash: &str, config: ModelConfig) {
        let tx = self.tx.as_ref().unwrap().clone();
        tx.send((id, hash.to_owned(), config)).await.unwrap();
    }

    pub async fn add_model(&mut self, id: Uuid, model: Model) {
        let mut project = self.projects.take(&id).unwrap();
        project.add_model(model);
        self.projects.insert(project);
        self.write().await.unwrap();
    }

    pub async fn remove_model(&mut self, id: Uuid, model_id: Uuid) {
        let mut project = self.projects.take(&id).unwrap();
        let model = project.remove_model(model_id);
        match model {
            Some(model) => {
                tokio::fs::remove_dir_all(model.get_path()).await.unwrap();
            }
            None => {}
        }
        self.projects.insert(project);
        self.write().await.unwrap();
    }
}
