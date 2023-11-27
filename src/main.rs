#[macro_use]
extern crate command_macros;
use axum::{
    body::StreamBody,
    debug_handler,
    extract::{Path, State},
    http::{header, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use chrono::prelude::*;
use clap::Parser;
use hyper::body::Bytes;
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
            let mut state = state_clone.lock().await;
            let project = state.get_project(&id).is_some();
            if !project {
                tracing::error!("Project {} not found", id);
                state.set_training(&id, &hash, false);
                continue;
            }
            let datahandle = state
                .get_project(&id)
                .unwrap()
                .get_datahandle(&hash)
                .is_some();
            if !datahandle {
                tracing::error!("Datahandle {} not found", hash);
                state.set_training(&id, &hash, false);
                continue;
            }
            drop(state);
            tracing::info!("Building model {} for project {}", model_config.name, id);
            let model = types::Model::new(state_clone.clone(), id, &hash, model_config).await;
            let mut state = state_clone.lock().await;
            state.add_model(id, model).await;
            state.set_training(&id, &hash, false);
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
        .route("/download", get(get_data))
        .route("/models", get(get_models))
        .route("/add", post(add_model));

    let model = Router::new()
        .route("/", get(get_model).delete(delete_model))
        .route("/predict", post(model_predict))
        .route("/metrics", get(get_model_metrics))
        .route("/download", get(get_model_file));

    let project = Router::new()
        .route("/", get(project_info).delete(delete_project))
        .route("/add", post(upload_data))
        .nest("/file/:hash", data)
        .nest("/model/:model", model);

    let app = Router::new()
        .route("/project", get(projects_list).post(new_project))
        .route("/clean", put(clean_data))
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
async fn clean_data(
    State(state): State<Arc<Mutex<types::AppState>>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let mut name = chrono::Utc::now().timestamp_millis().to_string();
    name.push_str(".json");
    let file = state.lock().await.get_path().join(name);
    tokio::fs::write(file.clone(), payload.to_string())
        .await
        .unwrap();
    let script = state
        .lock()
        .await
        .get_path()
        .join("../scripts/clean_data.py");
    cmd!(python3(script)(file.to_str().unwrap()))
        .status()
        .unwrap();
    let result = tokio::fs::read_to_string(file.clone()).await.unwrap();
    tokio::fs::remove_file(file).await.unwrap();
    let result: Value = serde_json::from_str(result.as_str()).unwrap();
    (StatusCode::OK, Json(result))
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
                "files": project.get_datahandles().iter().map(|d| {
                    let info = d.get_info();

                    let models = project
                        .get_models(&d.get_hash())
                        .iter()
                        .map(|model| model.get_info())
                        .map(|model| {
                            (model.get("id").unwrap().as_str().unwrap().to_string(), model.get("name").unwrap().as_str().unwrap().to_string())
                        })
                        .collect::<Vec<_>>();

                    json!({
                        "name": info.get("name").unwrap(),
                        "size": info.get("size").unwrap(),
                        "content_type": info.get("content_type").unwrap(),
                        "hash": info.get("hash").unwrap(),
                        "shape": info.get("shape").unwrap(),
                        "features": info.get("features").unwrap(),
                        "target": info.get("target").unwrap(),
                        "multi_class": info.get("multi_class").unwrap(),
                        "head": info.get("head").unwrap(),
                        "empty": info.get("empty").unwrap(),
                        "scale": info.get("scale").unwrap(),
                        "categorical": info.get("categorical").unwrap(),
                        "models": models,
                        "training": state.get_training(&id, &d.get_hash()),
                    })
                }).collect::<Vec<_>>(),
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
            let models = project
                .get_models(&hash)
                .iter()
                .map(|model| model.get_info())
                .map(|model| {
                    (
                        model.get("id").unwrap().as_str().unwrap().to_string(),
                        model.get("name").unwrap().as_str().unwrap().to_string(),
                    )
                })
                .collect::<Vec<_>>();
            let training = state.get_training(&id, &hash);
            match data {
                Some(data) => (
                    StatusCode::OK,
                    Json(json!({
                        "name": data.get_name(),
                        "size": data.get_size(),
                        "content_type": data.get_content_type(),
                        "hash": data.get_hash(),
                        "shape": data.get_shape(),
                        "features": data.get_features(),
                        "target": data.get_target(),
                        "multi_class": data.get_multi_class(),
                        "head": data.get_head(),
                        "empty": data.get_empty(),
                        "scale": data.get_scale(),
                        "categorical": data.get_categorical(),
                        "models": models,
                        "training": training,
                    })),
                ),
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
    Json(payload): Json<types::Upload>,
) -> (StatusCode, Json<Value>) {
    println!("{:?}", payload);
    let content = payload.file.content.clone().into_bytes();
    let data = Bytes::from(content);

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
                    let path = state.get_path().join(id.to_string());
                    let data_handle = types::DataHandle::new(
                        path,
                        payload.file.name,
                        payload.file.content_type,
                        data,
                        payload.target,
                        payload.empty,
                        payload.scale,
                        payload.categorical,
                    )
                    .await;
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
    State(statehandle): State<Arc<Mutex<types::AppState>>>,
    Path((id, hash)): Path<(Uuid, String)>,
    Json(mut config): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let mut state = statehandle.lock().await;
    let project = state.get_project(&id).is_some();
    if project {
        match state.get_project(&id).unwrap().get_datahandle(&hash) {
            Some(datahandle) => {
                let model_id = Uuid::new_v4();
                let mut name = chrono::Utc::now().timestamp_millis().to_string();
                name.push_str(".json");
                let file = state.get_path().join(name);
                match config {
                    Value::Object(ref mut map) => {
                        map.insert("id".to_string(), json!(model_id));
                        map.insert("project".to_string(), json!(id));
                        map.insert(
                            "file".to_string(),
                            json!(std::fs::canonicalize(datahandle.get_path()).unwrap()),
                        );
                        map.insert("target".to_string(), json!(datahandle.get_target()));
                        map.insert(
                            "multi_class".to_string(),
                            json!(datahandle.get_multi_class()),
                        );
                    }
                    _ => unreachable!(),
                }
                tokio::fs::write(file.clone(), config.to_string())
                    .await
                    .unwrap();
                let script = state.get_path().join("../scripts/train.py");
                state.set_training(&id, &hash, true);
                drop(state);
                cmd!(python3(script)(file.to_str().unwrap()))
                    .status()
                    .unwrap();
                let result = tokio::fs::read_to_string(file.clone()).await.unwrap();
                tokio::fs::remove_file(file).await.unwrap();
                let mut state = statehandle.lock().await;
                state
                    .add_model2(
                        hash.as_str(),
                        serde_json::from_str(result.as_str()).unwrap(),
                    )
                    .await;
                state.set_training(&id, &hash, false);
                (StatusCode::CREATED, Json(json!({ "id": model_id })))
                /* let config = match types::ModelConfig::from_json(datahandle, &config) {
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
                ) */
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
                    let res = model.predict2(input);
                    (StatusCode::OK, Json(res))
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
