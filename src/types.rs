use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::Metadata;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone)]
pub struct AppState {
    server_path: PathBuf,
    projects: HashSet<Uuid>,
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            server_path: PathBuf::new(),
            projects: HashSet::new(),
        }
    }
}

impl AppState {
    pub async fn new(server_path: PathBuf) -> Self {
        let mut state = AppState {
            server_path,
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

    pub fn get_projects(&self) -> HashSet<Uuid> {
        self.projects.clone()
    }

    pub async fn add(&mut self, id: Uuid) {
        self.projects.insert(id);
        self.write().await.unwrap();
    }

    pub async fn remove(&mut self, id: &Uuid) {
        self.projects.remove(id);
        self.write().await.unwrap();
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Project {
    id: Uuid,
    name: String,
    description: String,
    expires: i64,
}

impl Default for Project {
    fn default() -> Self {
        Project {
            id: Uuid::new_v4(),
            name: String::new(),
            description: String::new(),
            expires: 0,
        }
    }
}

impl Project {
    pub fn new(name: String, description: String, expires: i64) -> Self {
        Project {
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
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Model {
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

#[derive(Serialize, Deserialize, Clone)]
pub struct DataHandle {
    name: String,
    size: u64,
    content_type: String,
    shape: (u64, u64),
    features: Vec<String>,
    target: String,
    head: Vec<Vec<String>>,
    models: Vec<Model>,
}

impl Default for DataHandle {
    fn default() -> Self {
        Self {
            name: String::new(),
            size: 0,
            content_type: String::new(),
            shape: (0, 0),
            features: Vec::new(),
            target: String::new(),
            head: Vec::new(),
            models: Vec::new(),
        }
    }
}

impl DataHandle {
    pub fn new(path: &PathBuf, metadata: &Metadata, content_type: &str, target: &str) -> Self {
        let mut data_handle = Self::default();

        let mut csv_file = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .unwrap();
        let records = csv_file
            .records()
            .filter_map(|record| record.ok())
            .collect::<Vec<_>>();

        data_handle.name = path.file_name().unwrap().to_str().unwrap().to_string();
        data_handle.size = metadata.len();
        data_handle.content_type = content_type.to_string();
        data_handle.shape = (
            records.len() as u64,
            csv_file.headers().unwrap().len() as u64 - 1,
        );
        data_handle.features = csv_file
            .headers()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .filter(|s| s != target)
            .collect();
        data_handle.target = target.to_string();
        data_handle.head = records
            .into_iter()
            .take(5)
            .map(|record| record.iter().map(|s| s.to_string()).collect())
            .collect();
        data_handle.models = Vec::new();
        data_handle
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_size(&self) -> u64 {
        self.size
    }

    pub fn get_content_type(&self) -> String {
        self.content_type.clone()
    }

    pub fn get_shape(&self) -> (u64, u64) {
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

    pub fn get_models(&self) -> Vec<Model> {
        self.models.clone()
    }
}
