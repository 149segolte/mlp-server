use axum::{
    body::StreamBody,
    debug_handler,
    extract::{Multipart, Path, State},
    http::{header, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::prelude::*;
use clap::Parser;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::fs::{create_dir, remove_dir_all};
use tokio::io::BufReader;
use tokio::sync::Mutex;
use tokio_util::io::ReaderStream;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use uuid::Uuid;

pub mod types;

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, default_value = "8080")]
    port: u16,
    #[clap(short, long, default_value = "/")]
    route: String,
}

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // get args
    let args = Args::parse();

    // create State
    let path = std::env::current_dir().unwrap().join(".server");
    if !path.exists() {
        create_dir(&path).await.unwrap();
    }
    let state = Arc::new(Mutex::new(types::AppState::new(path).await));

    // create cors layer
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE])
        .allow_origin(Any);

    // build our application with a router
    let data = Router::new()
        .route("/", get(get_datahandle).delete(delete_datahandle))
        .route("/file", get(get_data));

    let project = Router::new()
        .route("/", get(project_info).delete(delete_project))
        .route("/upload", post(upload_data))
        .nest("/:hash", data);

    let app = Router::new()
        .route("/projects", get(projects_list))
        .route("/project", post(new_project))
        .nest("/project/:id", project);

    // run it on localhost
    let routing = Router::new()
        .nest(&args.route, app)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let addr = format!("[::]:{}", args.port).parse().unwrap();
    let server = axum::Server::bind(&addr);
    tracing::debug!("Listening on {}", addr);
    server.serve(routing.into_make_service()).await.unwrap();
}

async fn validate_project(
    state: Arc<Mutex<types::AppState>>,
    id: Uuid,
) -> Result<types::Project, &'static str> {
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
            let project = serde_json::from_str::<types::Project>(&config);
            match project {
                Ok(project) => {
                    let now = Utc::now().timestamp();
                    if project.get_expires() < now {
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

async fn remove_project(state: Arc<Mutex<types::AppState>>, id: Uuid) {
    let mut state = state.lock().await;
    state.remove(&id).await;
    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        remove_dir_all(project_path).await.unwrap();
    }
}

#[debug_handler]
async fn new_project(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Json(new): Json<Value>,
) -> (StatusCode, Json<Uuid>) {
    if new.get("name").is_none() {
        return (StatusCode::BAD_REQUEST, Json(Uuid::nil()));
    }
    if new.get("description").is_none() {
        return (StatusCode::BAD_REQUEST, Json(Uuid::nil()));
    }
    let project = types::Project::new(
        new.get("name").unwrap().as_str().unwrap().to_string(),
        new.get("description")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string(),
        Utc::now().timestamp() + 60 * 60 * 24,
    );
    let mut state = state.lock().await;
    let project_path = state.get_path().join(project.get_id().to_string());
    create_dir(&project_path).await.unwrap();
    let project_json = serde_json::to_string(&project).unwrap();
    tokio::fs::write(project_path.join("project.json"), project_json)
        .await
        .unwrap();
    state.add(project.get_id()).await;
    (StatusCode::CREATED, Json(project.get_id()))
}

#[debug_handler]
async fn projects_list(
    State(state): State<Arc<Mutex<types::AppState>>>,
) -> Json<Vec<types::Project>> {
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

#[debug_handler]
async fn project_info(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    let project = validate_project(state.clone(), id).await;
    match project {
        Ok(project) => {
            let project_path = state.lock().await.get_path().join(id.to_string());
            let data_file = tokio::fs::read_to_string(project_path.join("data.json")).await;
            let data_handles = match data_file {
                Ok(file) => {
                    let data = serde_json::from_str::<HashMap<String, types::DataHandle>>(&file);
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
                    "id": project.get_id(),
                    "name": project.get_name(),
                    "description": project.get_description(),
                    "expires": project.get_expires(),
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
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    remove_project(state.clone(), id).await;
    (StatusCode::OK, Json(json!({})))
}

async fn validate_datahandle(
    state: Arc<Mutex<types::AppState>>,
    id: Uuid,
    hash: String,
) -> Result<types::DataHandle, &'static str> {
    let project = validate_project(state.clone(), id).await;
    match project {
        Ok(_) => {
            let project_path = state.lock().await.get_path().join(id.to_string());
            let data_file = tokio::fs::read_to_string(project_path.join("data.json")).await;
            match data_file {
                Ok(file) => {
                    let data_handle =
                        serde_json::from_str::<HashMap<String, types::DataHandle>>(&file);
                    match data_handle {
                        Ok(data_handle) => {
                            if data_handle.contains_key(&hash) {
                                let data = data_handle.get(&hash).unwrap();
                                let reader = BufReader::new(
                                    tokio::fs::File::open(project_path.join(&data.get_name()))
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
    state: Arc<Mutex<types::AppState>>,
    id: Uuid,
    hash: String,
) -> Result<(), &'static str> {
    let state = state.lock().await;
    let project_path = state.get_path().join(id.to_string());
    if project_path.exists() {
        let mut data_file = match tokio::fs::read_to_string(project_path.join("data.json")).await {
            Ok(file) => {
                let data = serde_json::from_str::<HashMap<String, types::DataHandle>>(&file);
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
            tokio::fs::remove_file(project_path.join(&data.get_name()))
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
async fn upload_data(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path(id): Path<Uuid>,
    mut payload: Multipart,
) -> (StatusCode, Json<Value>) {
    let project = validate_project(state.clone(), id).await;
    match project {
        Ok(_) => {
            let project_path = state.lock().await.get_path().join(id.to_string());
            let mut data_handles = match tokio::fs::read_to_string(project_path.join("data.json"))
                .await
            {
                Ok(file) => {
                    let data = serde_json::from_str::<HashMap<String, types::DataHandle>>(&file);
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
                    let data_handle = types::DataHandle::new(
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
    State(state): State<Arc<Mutex<types::AppState>>>,
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
    State(state): State<Arc<Mutex<types::AppState>>>,
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
    State(state): State<Arc<Mutex<types::AppState>>>,
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
        .join(&datahandle.get_name());
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
