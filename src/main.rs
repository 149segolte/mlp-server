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
use std::sync::Arc;
use tokio::fs::create_dir;
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

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);

    // create State
    let path = std::env::current_dir().unwrap().join(".server");
    if !path.exists() {
        create_dir(&path).await.unwrap();
    }
    let state = Arc::new(Mutex::new(types::AppState::new(path, tx.clone()).await));

    // setup model build queue
    let state_clone = state.clone();
    tokio::spawn(async move {
        let state_clone = state_clone.clone();
        while let Some((id, hash, model_config)) = rx.recv().await {
            let state = state_clone.lock().await;
            let project = state.get_project(&id).is_some();
            if !project {
                tracing::error!("Project {} not found", id);
                continue;
            }
            let datahandle = state
                .get_project(&id)
                .unwrap()
                .get_datahandle(&hash)
                .is_some();
            if !datahandle {
                tracing::error!("Datahandle {} not found", hash);
                continue;
            }
            drop(state);
            tracing::info!("Building model {} for project {}", model_config.name, id);
            let model = types::Model::new(state_clone.clone(), id, &hash, model_config).await;
            let mut state = state_clone.lock().await;
            state.add_model(id, model).await;
        }
    });
    //
    // create cors layer
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE])
        .allow_origin(Any);

    // build our application with a router
    let data = Router::new()
        .route("/", get(get_datahandle).delete(delete_datahandle))
        .route("/file", get(get_data))
        .route("/models", get(get_models))
        .route("/add", post(add_model));

    let model = Router::new()
        .route("/", get(get_model).delete(delete_model))
        .route("/predict", post(model_predict))
        .route("/metrics", get(get_model_metrics))
        .route("/file", get(get_model_file));

    let project = Router::new()
        .route("/", get(project_info).delete(delete_project))
        .route("/data", post(upload_data))
        .nest("/data/:hash", data)
        .nest("/model/:model", model);

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

#[debug_handler]
async fn projects_list(State(state): State<Arc<Mutex<types::AppState>>>) -> Json<Value> {
    let state = state.lock().await;
    let projects = state
        .get_projects()
        .iter()
        .map(|p| p.get_info())
        .collect::<Vec<_>>();
    Json(json!({
        "projects": projects,
    }))
}

#[debug_handler]
async fn new_project(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Json(new): Json<Value>,
) -> (StatusCode, Json<Value>) {
    if new.get("name").is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Missing name" })),
        );
    }
    if new.get("description").is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Missing description" })),
        );
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
    let id = state.add(project).await;
    (StatusCode::CREATED, Json(json!({ "id": id })))
}

#[debug_handler]
async fn project_info(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path(id): Path<Uuid>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => (
            StatusCode::OK,
            Json(json!({
                "name": project.get_name(),
                "description": project.get_description(),
                "expires": project.get_expires(),
                "data": project.get_datahandles().iter().map(|d| d.get_info()).collect::<Vec<_>>()
            })),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        ),
    }
}

#[debug_handler]
async fn delete_project(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path(id): Path<Uuid>,
) -> StatusCode {
    let mut state = state.lock().await;
    state.remove(id).await;
    StatusCode::OK
}

#[debug_handler]
async fn get_datahandle(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let data = project.get_datahandle(&hash);
            match data {
                Some(data) => (StatusCode::OK, Json(data.get_info())),
                None => (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "Data not found" })),
                ),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        ),
    }
}

#[debug_handler]
async fn upload_data(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path(id): Path<Uuid>,
    mut payload: Multipart,
) -> (StatusCode, Json<Value>) {
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
    if fields.len() != 2 || fields.iter().find(|field| field.0 == "target").is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Missing data or target" })),
        );
    }

    if let Some(file) = fields.iter().find(|field| field.0 != "target") {
        let name = file.0.clone();
        let content_type = file.1.clone();
        let data = file.2.clone();

        let mut state = state.lock().await;
        let project = state.get_project(&id);
        match project {
            Some(project) => {
                let hash = sha256::digest(data.to_vec());
                let data_handle = project.get_datahandle(hash.as_str());
                match data_handle {
                    Some(_) => (
                        StatusCode::OK,
                        Json(json!({ "status": "Already exists", "hash": hash })),
                    ),
                    None => {
                        let target = std::str::from_utf8(
                            fields
                                .iter()
                                .find(|field| field.0 == "target")
                                .unwrap()
                                .2
                                .to_vec()
                                .as_slice(),
                        )
                        .unwrap()
                        .to_string();
                        let path = state.get_path().join(id.to_string());
                        let data_handle =
                            types::DataHandle::new(path, name, content_type, data, target).await;
                        match data_handle {
                            Ok(data_handle) => {
                                state.add_datahandle(id, data_handle).await;
                                (StatusCode::CREATED, Json(json!({ "hash": hash })))
                            }
                            Err(err) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": err })),
                            ),
                        }
                    }
                }
            }
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "Project not found" })),
            ),
        }
    } else {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid number of fields" })),
        )
    }
}

#[debug_handler]
async fn delete_datahandle(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> StatusCode {
    let mut state = state.lock().await;
    state.remove_datahandle(id, &hash).await;
    StatusCode::OK
}

#[debug_handler]
async fn get_data(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> Response {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let data = project.get_datahandle(&hash);
            match data {
                Some(data) => {
                    let file = match tokio::fs::File::open(data.get_path()).await {
                        Ok(file) => file,
                        Err(err) => {
                            return (
                                StatusCode::NOT_FOUND,
                                serde_json::to_string(&json!({ "error": err.to_string() }))
                                    .unwrap(),
                            )
                                .into_response()
                        }
                    };
                    let stream = ReaderStream::new(file);
                    let body = StreamBody::new(stream);
                    let content_type = data.get_content_type();

                    (
                        StatusCode::OK,
                        [
                            (header::CONTENT_TYPE, content_type.as_str()),
                            (header::CONTENT_DISPOSITION, data.get_name().as_str()),
                        ],
                        body,
                    )
                        .into_response()
                }
                None => (
                    StatusCode::NOT_FOUND,
                    serde_json::to_string(&json!({ "error": "Data not found" })).unwrap(),
                )
                    .into_response(),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            serde_json::to_string(&json!({ "error": "Project not found" })).unwrap(),
        )
            .into_response(),
    }
}

#[debug_handler]
async fn get_models(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let models = project
                .get_models(&hash)
                .iter()
                .map(|model| model.get_info())
                .collect::<Vec<_>>();
            (StatusCode::OK, Json(json!({ "models": models })))
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        ),
    }
}

#[debug_handler]
async fn add_model(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
    Json(config): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id).is_some();
    if project {
        match state.get_project(&id).unwrap().get_datahandle(&hash) {
            Some(datahandle) => {
                let config = match types::ModelConfig::from_json(datahandle, &config) {
                    Ok(config) => config,
                    Err(err) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({ "error": err.to_string() })),
                        )
                    }
                };
                state.add_model_queue(id, &hash, config).await;
                (
                    StatusCode::CREATED,
                    Json(json!({ "status": "Added to queue" })),
                )
            }
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "Data not found" })),
            ),
        }
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        )
    }
}

#[debug_handler]
async fn get_model(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, model_id)): Path<(Uuid, Uuid)>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let model = project.get_model(model_id);
            match model {
                Some(model) => (StatusCode::OK, Json(model.get_overview().await)),
                None => (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "Model not found" })),
                ),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        ),
    }
}

#[debug_handler]
async fn delete_model(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, model_id)): Path<(Uuid, Uuid)>,
) -> StatusCode {
    let mut state = state.lock().await;
    state.remove_model(id, model_id).await;
    StatusCode::OK
}

#[debug_handler]
async fn model_predict(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, model_id)): Path<(Uuid, Uuid)>,
    Json(data): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let model = project.get_model(model_id);
            match model {
                Some(model) => {
                    let input = match data.get("input") {
                        Some(input) => input,
                        None => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": "Missing input" })),
                            )
                        }
                    };
                    let data: Vec<Vec<f32>> = serde_json::from_value(input.clone()).unwrap();
                    let data = ndarray::Array::from_shape_vec(
                        (data.len(), data[0].len()),
                        data.into_iter().flatten().collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let res = model.predict(&data);
                    let val = serde_json::to_value(res).unwrap();
                    (StatusCode::OK, Json(json!(val)))
                }
                None => (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "Model not found" })),
                ),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        ),
    }
}

#[debug_handler]
async fn get_model_metrics(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, model_id)): Path<(Uuid, Uuid)>,
) -> (StatusCode, Json<Value>) {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let model = project.get_model(model_id);
            match model {
                Some(model) => (
                    StatusCode::OK,
                    Json(serde_json::to_value(model.get_metrics()).unwrap()),
                ),
                None => (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "Model not found" })),
                ),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Project not found" })),
        ),
    }
}

#[debug_handler]
async fn get_model_file(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Path((id, model_id)): Path<(Uuid, Uuid)>,
) -> Response {
    let state = state.lock().await;
    let project = state.get_project(&id);
    match project {
        Some(project) => {
            let model = project.get_model(model_id);
            match model {
                Some(model) => {
                    let file = match tokio::fs::File::open(model.get_path().join("model.bin")).await
                    {
                        Ok(file) => file,
                        Err(err) => {
                            return (
                                StatusCode::NOT_FOUND,
                                serde_json::to_string(&json!({ "error": err.to_string() }))
                                    .unwrap(),
                            )
                                .into_response()
                        }
                    };
                    let stream = ReaderStream::new(file);
                    let body = StreamBody::new(stream);
                    let content_type = "application/octet-stream";

                    (
                        StatusCode::OK,
                        [
                            (header::CONTENT_TYPE, content_type),
                            (header::CONTENT_DISPOSITION, model.get_name().as_str()),
                        ],
                        body,
                    )
                        .into_response()
                }
                None => (
                    StatusCode::NOT_FOUND,
                    serde_json::to_string(&json!({ "error": "Model not found" })).unwrap(),
                )
                    .into_response(),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            serde_json::to_string(&json!({ "error": "Project not found" })).unwrap(),
        )
            .into_response(),
    }
}
