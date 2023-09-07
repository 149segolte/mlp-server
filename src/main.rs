use axum::debug_handler;
use axum::http::StatusCode;
use axum::{
    extract::{Path, State},
    routing::delete,
    routing::get,
    routing::post,
    Json, Router,
};
use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::{create_dir, remove_dir_all};
use tokio::sync::Mutex;
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
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // run it with hyper on localhost
    let addr = ([127, 0, 0, 1], port).into();
    let server = axum::Server::bind(&addr);
    tracing::debug!("Listening on {}", addr);
    server.serve(app.into_make_service()).await.unwrap();
}

#[debug_handler]
async fn projects(State(state): State<Arc<Mutex<AppState>>>) -> Json<Vec<Project>> {
    let mut state = state.lock().await;
    let mut projects = Vec::new();
    let uuids = state.get_projects();
    for id in uuids.iter() {
        let project_path = state.get_path().join(id.to_string());
        if project_path.exists() {
            let config = tokio::fs::read_to_string(project_path.join("project.json")).await;
            match config {
                Ok(config) => {
                    let project = serde_json::from_str(&config);
                    match project {
                        Ok(project) => projects.push(project),
                        Err(_) => {
                            tracing::error!("Error parsing project.json for {}", id);
                            state.remove(id).await;
                        }
                    }
                }
                Err(_) => {
                    tracing::error!("Error reading project.json for {}", id);
                    state.remove(id).await;
                }
            }
        } else {
            tracing::error!("Project {} does not exist", id);
            state.remove(id).await;
        }
    }
    Json(projects)
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
) -> (StatusCode, Json<Value>) {
    if new.name.is_empty() {
        tracing::error!("Project name cannot be empty");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Project name cannot be empty"})),
        );
    }
    if new.description.is_empty() {
        tracing::error!("Project description cannot be empty");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Project description cannot be empty"})),
        );
    }

    let mut state = state.lock().await;
    let mut id;
    loop {
        id = Uuid::new_v4();
        if !state.get_projects().contains(&id) {
            break;
        }
    }

    let project = Project {
        id,
        name: new.name.clone(),
        description: new.description.clone(),
        expires: Utc::now().timestamp() + 60 * 60 * 24,
    };

    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        tracing::error!("Project {} exists but is not in config", id);
        let remove = remove_dir_all(project_path.clone()).await;
        match remove {
            Ok(_) => (),
            Err(_) => {
                tracing::error!("Error removing project {}", id);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Internal server error"})),
                );
            }
        }
    }

    let create = create_dir(project_path.clone()).await;
    match create {
        Ok(_) => (),
        Err(_) => {
            tracing::error!("Error creating project {}", id);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            );
        }
    }
    let project_json = serde_json::to_string(&project);
    match project_json {
        Ok(project_json) => {
            tokio::fs::write(project_path.join("project.json"), project_json)
                .await
                .unwrap();
            state.add(id).await;
            (StatusCode::CREATED, Json(json!({"id": id})))
        }
        Err(_) => {
            tracing::error!("Error serializing project {}", id);
            let remove = remove_dir_all(project_path.clone()).await;
            match remove {
                Ok(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Internal server error"})),
                ),
                Err(_) => {
                    tracing::error!("Error removing project {}", id);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": "Internal server error"})),
                    )
                }
            }
        }
    }
}

#[debug_handler]
async fn get_project(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    let mut state = state.lock().await;
    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        let config = tokio::fs::read_to_string(project_path.join("project.json")).await;
        match config {
            Ok(config) => {
                let project = serde_json::from_str::<Project>(&config);
                match project {
                    Ok(project) => (StatusCode::OK, Json(json!(project))),
                    Err(_) => {
                        tracing::error!("Error parsing project.json for {}", id);
                        state.remove(&id).await;
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error": "Internal server error"})),
                        )
                    }
                }
            }
            Err(_) => {
                tracing::error!("Error reading project.json for {}", id);
                state.remove(&id).await;
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Internal server error"})),
                )
            }
        }
    } else {
        tracing::error!("Project {} does not exist", id);
        state.remove(&id).await;
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Project does not exist"})),
        )
    }
}

#[debug_handler]
async fn delete_project(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    let mut state = state.lock().await;
    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        let remove = remove_dir_all(project_path.clone()).await;
        match remove {
            Ok(_) => {
                state.remove(&id).await;
                (StatusCode::OK, Json(json!({})))
            }
            Err(_) => {
                tracing::error!("Error removing project {}", id);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Internal server error"})),
                )
            }
        }
    } else {
        tracing::error!("Project {} does not exist", id);
        state.remove(&id).await;
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Project does not exist"})),
        )
    }
}
