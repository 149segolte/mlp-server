use axum::body::StreamBody;
use axum::debug_handler;
use axum::extract::Multipart;
use axum::response::{IntoResponse, Response};
use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    routing::{delete, get, post},
    Json, Router,
};
use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::Metadata;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::{create_dir, remove_dir_all};
use tokio::io::BufReader;
use tokio::sync::Mutex;
use tokio_util::io::ReaderStream;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone)]
struct AppState {
    server_path: PathBuf,
    port: u16,
    projects: HashSet<Uuid>,
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            server_path: PathBuf::new(),
            port: 3000,
            projects: HashSet::new(),
        }
    }
}

impl AppState {
    async fn new(server_path: PathBuf) -> Self {
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
        self.port = config.port;
        Ok(())
    }

    fn get_path(&self) -> PathBuf {
        self.server_path.clone()
    }

    fn get_port(&self) -> u16 {
        self.port
    }

    async fn set_port(&mut self, port: u16) {
        self.port = port;
        self.write().await.unwrap();
    }

    fn get_projects(&self) -> HashSet<Uuid> {
        self.projects.clone()
    }

    async fn add(&mut self, id: Uuid) {
        self.projects.insert(id);
        self.write().await.unwrap();
    }

    async fn remove(&mut self, id: &Uuid) {
        self.projects.remove(id);
        self.write().await.unwrap();
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Project {
    id: Uuid,
    name: String,
    description: String,
    expires: i64,
}

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // create State
    let path = std::env::current_dir().unwrap().join(".server");
    if !path.exists() {
        create_dir(&path).await.unwrap();
    }
    let state = Arc::new(Mutex::new(AppState::new(path).await));
    let port = state.lock().await.get_port();

    // build our application with a router
    let app = Router::new()
        .route("/projects/list", get(projects))
        .route("/projects/new", post(new_project))
        .route("/projects/:id", get(get_project))
        .route("/projects/:id", delete(delete_project))
        .route("/projects/:id/new", post(new_data))
        .route("/projects/:id/:hash", get(get_datahandle))
        .route("/projects/:id/:hash", delete(delete_datahandle))
        .route("/projects/:id/:hash/file", get(get_data))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // run it with hyper on localhost
    let addr = ([127, 0, 0, 1], port).into();
    let server = axum::Server::bind(&addr);
    tracing::debug!("Listening on {}", addr);
    server.serve(app.into_make_service()).await.unwrap();
}

async fn validate_project(state: Arc<Mutex<AppState>>, id: Uuid) -> Result<Project, &'static str> {
    let state = state.lock().await;
    if !state.get_projects().contains(&id) {
        return Err("Project not found in config");
    }
    let project_path = state.get_path().join(id.to_string());
    if !project_path.exists() {
        return Err("Project not found on disk");
    }
    let config = tokio::fs::read_to_string(project_path.join("project.json")).await;
    match config {
        Ok(config) => {
            let project = serde_json::from_str::<Project>(&config);
            match project {
                Ok(project) => {
                    let now = Utc::now().timestamp();
                    if project.expires < now {
                        return Err("Project has expired");
                    }
                    Ok(project.clone())
                }
                Err(_) => Err("Error parsing project.json"),
            }
        }
        Err(_) => Err("Error reading project.json"),
    }
}

async fn remove_project(state: Arc<Mutex<AppState>>, id: Uuid) {
    let mut state = state.lock().await;
    state.remove(&id).await;
    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        remove_dir_all(project_path).await.unwrap();
    }
}

#[derive(Deserialize)]
struct NewProject {
    name: String,
    description: String,
}

#[debug_handler]
async fn new_project(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(new): Json<NewProject>,
) -> (StatusCode, Json<Uuid>) {
    if new.name.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(Uuid::nil()));
    }
    if new.description.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(Uuid::nil()));
    }
    let id = Uuid::new_v4();
    remove_project(state.clone(), id).await;
    let mut state = state.lock().await;
    let project_path = state.get_path().join(id.to_string());
    create_dir(&project_path).await.unwrap();
    let project = Project {
        id,
        name: new.name,
        description: new.description,
        expires: Utc::now().timestamp() + 60 * 60 * 24,
    };
    let project_json = serde_json::to_string(&project).unwrap();
    tokio::fs::write(project_path.join("project.json"), project_json)
        .await
        .unwrap();
    state.add(id).await;
    (StatusCode::CREATED, Json(id))
}

#[debug_handler]
async fn projects(State(state): State<Arc<Mutex<AppState>>>) -> Json<Vec<Project>> {
    let mut projects = Vec::new();
    let uuids = state.lock().await.get_projects();
    for id in uuids.iter() {
        let project = validate_project(state.clone(), *id).await;
        match project {
            Ok(project) => projects.push(project),
            Err(err) => {
                tracing::error!("Error validating project {}: {}", id, err);
                remove_project(state.clone(), *id).await;
            }
        }
    }
    Json(projects)
}

#[derive(Serialize, Deserialize, Clone)]
enum Model {
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
struct DataHandle {
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
    fn new(path: &PathBuf, metadata: &Metadata, content_type: &str, target: &str) -> Self {
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
}

#[debug_handler]
async fn get_project(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    let project = validate_project(state.clone(), id).await;
    match project {
        Ok(project) => {
            let project_path = state.lock().await.get_path().join(id.to_string());
            let data_file = tokio::fs::read_to_string(project_path.join("data.json")).await;
            let data_handles = match data_file {
                Ok(file) => {
                    let data = serde_json::from_str::<HashMap<String, DataHandle>>(&file);
                    match data {
                        Ok(data_handles) => data_handles,
                        Err(_) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": "Error parsing data.json" })),
                            )
                        }
                    }
                }
                Err(_) => HashMap::new(),
            };
            (
                StatusCode::OK,
                Json(json!({
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "expires": project.expires,
                    "data": data_handles,
                })),
            )
        }
        Err(err) => {
            tracing::error!("Error validating project {}: {}", id, err);
            remove_project(state.clone(), id).await;
            (StatusCode::NOT_FOUND, Json(json!({})))
        }
    }
}

#[debug_handler]
async fn delete_project(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    remove_project(state.clone(), id).await;
    (StatusCode::OK, Json(json!({})))
}

async fn validate_datahandle(
    state: Arc<Mutex<AppState>>,
    id: Uuid,
    hash: String,
) -> Result<DataHandle, &'static str> {
    let project = validate_project(state.clone(), id).await;
    match project {
        Ok(_) => {
            let project_path = state.lock().await.get_path().join(id.to_string());
            let data_file = tokio::fs::read_to_string(project_path.join("data.json")).await;
            match data_file {
                Ok(file) => {
                    let data_handle = serde_json::from_str::<HashMap<String, DataHandle>>(&file);
                    match data_handle {
                        Ok(data_handle) => {
                            if data_handle.contains_key(&hash) {
                                let data = data_handle.get(&hash).unwrap();
                                let reader = BufReader::new(
                                    tokio::fs::File::open(project_path.join(&data.name))
                                        .await
                                        .unwrap(),
                                );
                                if hash
                                    == sha256::async_calc(reader, openssl::sha::Sha256::new())
                                        .await
                                        .unwrap()
                                {
                                    Ok(data.clone())
                                } else {
                                    Err("Hash mismatch")
                                }
                            } else {
                                Err("Data not found")
                            }
                        }
                        Err(_) => Err("Error parsing data.json"),
                    }
                }
                Err(_) => Err("Error reading data.json"),
            }
        }
        Err(err) => {
            tracing::error!("Error validating project {}: {}", id, err);
            remove_project(state.clone(), id).await;
            Err("Project not found")
        }
    }
}

async fn remove_datahandle(
    state: Arc<Mutex<AppState>>,
    id: Uuid,
    hash: String,
) -> Result<(), &'static str> {
    let state = state.lock().await;
    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        let mut data_file = match tokio::fs::read_to_string(project_path.join("data.json")).await {
            Ok(file) => {
                let data = serde_json::from_str::<HashMap<String, DataHandle>>(&file);
                match data {
                    Ok(data) => data,
                    Err(_) => {
                        tracing::error!("Error parsing data.json");
                        return Err("Error parsing data.json");
                    }
                }
            }
            Err(_) => {
                tracing::error!("Error reading data.json");
                return Err("Error reading data.json");
            }
        };
        if data_file.contains_key(&hash) {
            let data = data_file.get(&hash).unwrap();
            tokio::fs::remove_file(project_path.join(&data.name))
                .await
                .unwrap();
            data_file.remove(&hash);
            let data_file = serde_json::to_string(&data_file).unwrap();
            tokio::fs::write(project_path.join("data.json"), data_file)
                .await
                .unwrap();
            Ok(())
        } else {
            tracing::error!("Data not found");
            Err("Data not found")
        }
    } else {
        tracing::error!("Project not found");
        Err("Project not found")
    }
}

#[debug_handler]
async fn new_data(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(id): Path<Uuid>,
    mut payload: Multipart,
) -> (StatusCode, Json<Value>) {
    let project = validate_project(state.clone(), id).await;
    match project {
        Ok(_) => {
            let project_path = state.lock().await.get_path().join(id.to_string());
            let mut data_handles =
                match tokio::fs::read_to_string(project_path.join("data.json")).await {
                    Ok(file) => {
                        let data = serde_json::from_str::<HashMap<String, DataHandle>>(&file);
                        match data {
                            Ok(data) => data,
                            Err(_) => {
                                return (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(json!({ "error": "Error parsing data.json" })),
                                )
                            }
                        }
                    }
                    Err(_) => HashMap::new(),
                };

            let mut fields = Vec::new();
            while let Some(field) = payload.next_field().await.unwrap() {
                let name = field.name().unwrap_or_default().to_string();
                if name == "data" {
                    let file_name = field.file_name().unwrap_or_default().to_string();
                    let content_type = field.content_type().unwrap_or_default().to_string();
                    let data = field.bytes().await.unwrap();
                    fields.push((file_name, content_type, data));
                } else if name == "target" {
                    fields.push((
                        "target".to_string(),
                        "".to_string(),
                        field.bytes().await.unwrap(),
                    ));
                }
            }
            if fields.len() != 2 {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "Invalid number of fields", "fields": fields.len() })),
                );
            }

            if let Some(file) = fields.iter().find(|field| field.0 != "target") {
                let name = file.0.clone();
                let content_type = file.1.clone();
                let data = file.2.clone();

                let hash =
                    sha256::async_calc(BufReader::new(data.as_ref()), openssl::sha::Sha256::new())
                        .await
                        .unwrap();

                if data_handles.contains_key(&hash) {
                    (StatusCode::FOUND, Json(json!(hash)))
                } else {
                    let path = project_path.join(&name);
                    tokio::fs::write(&path, data).await.unwrap();
                    let metadata = tokio::fs::metadata(project_path.join(&name)).await.unwrap();
                    let data_handle = DataHandle::new(
                        &path,
                        &metadata,
                        &content_type,
                        std::str::from_utf8(
                            &fields.iter().find(|field| field.0 == "target").unwrap().2,
                        )
                        .unwrap(),
                    );
                    data_handles.insert(hash.clone(), data_handle);
                    let data_file = serde_json::to_string(&data_handles).unwrap();
                    tokio::fs::write(project_path.join("data.json"), data_file)
                        .await
                        .unwrap();
                    (StatusCode::CREATED, Json(json!(hash)))
                }
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "Invalid data field" })),
                )
            }
        }
        Err(err) => {
            tracing::error!("Error validating project {}: {}", id, err);
            remove_project(state.clone(), id).await;
            (StatusCode::NOT_FOUND, Json(json!({})))
        }
    }
}

#[debug_handler]
async fn get_datahandle(
    State(state): State<Arc<Mutex<AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> (StatusCode, Json<Value>) {
    let datahandle = validate_datahandle(state.clone(), id, hash).await;
    match datahandle {
        Ok(datahandle) => (StatusCode::OK, Json(json!(datahandle))),
        Err(err) => (StatusCode::NOT_FOUND, Json(json!({ "error": err }))),
    }
}

#[debug_handler]
async fn delete_datahandle(
    State(state): State<Arc<Mutex<AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> (StatusCode, Json<Value>) {
    let datahandle = remove_datahandle(state.clone(), id, hash).await;
    match datahandle {
        Ok(_) => (StatusCode::OK, Json(json!({}))),
        Err(err) => (StatusCode::NOT_FOUND, Json(json!({ "error": err }))),
    }
}

#[debug_handler]
async fn get_data(
    State(state): State<Arc<Mutex<AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> Response {
    let datahandle = match validate_datahandle(state.clone(), id, hash).await {
        Ok(datahandle) => datahandle,
        Err(err) => {
            return (StatusCode::NOT_FOUND, format!("Data not found: {}", err)).into_response()
        }
    };
    let path = state
        .lock()
        .await
        .get_path()
        .join(id.to_string())
        .join(&datahandle.name);
    let file = match tokio::fs::File::open(&path).await {
        Ok(file) => file,
        Err(err) => {
            return (StatusCode::NOT_FOUND, format!("File not found: {}", err)).into_response()
        }
    };
    let stream = ReaderStream::new(file);
    let body = StreamBody::new(stream);
    let content_type = match mime_guess::from_path(&path).first_raw() {
        Some(mime) => mime,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "MIME Type couldn't be determined".to_string(),
            )
                .into_response()
        }
    };

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CONTENT_DISPOSITION,
                &format!("attachment; filename={:?}", path.file_name().unwrap()),
            ),
        ],
        body,
    )
        .into_response()
}
