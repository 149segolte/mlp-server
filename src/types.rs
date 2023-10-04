use hyper::body::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Borrow;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tokio::fs::create_dir_all;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
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

#[derive(Serialize, Deserialize, Clone, Debug, Eq, Default)]
pub struct DataHandle {
    path: PathBuf,
    content_type: String,
    hash: String,
    shape: (u64, u64),
    features: Vec<String>,
    target: String,
    head: Vec<Vec<String>>,
    models: Vec<Model>,
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
                let mut csv_file = csv::Reader::from_path(&data_handle.path).unwrap();
                let header = csv_file.headers().unwrap().clone();
                let records = csv_file
                    .into_records()
                    .filter_map(|r| r.ok())
                    .collect::<Vec<_>>();

                data_handle.shape = (
                    records.len() as u64,
                    header.len() as u64 - 1, // -1 for target
                );
                data_handle.features = header
                    .iter()
                    .map(|s| s.to_string())
                    .filter(|s| s != &target)
                    .collect();
                data_handle.target = target.to_string();
                data_handle.head = records
                    .into_iter()
                    .take(5)
                    .map(|record| record.iter().map(|s| s.to_string()).collect())
                    .collect();
                data_handle.models = Vec::new();
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
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq)]
pub struct Project {
    id: Uuid,
    name: String,
    description: String,
    expires: i64,
    data: HashSet<DataHandle>,
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

impl Default for Project {
    fn default() -> Self {
        Project {
            id: Uuid::new_v4(),
            name: String::new(),
            description: String::new(),
            expires: 0,
            data: HashSet::new(),
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
}

#[derive(Deserialize, Serialize, Clone, Default, Debug)]
pub struct AppState {
    server_path: PathBuf,
    projects: HashSet<Project>,
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
                self.projects.insert(project);
                self.write().await.unwrap();
            }
            None => {}
        }
    }
}
