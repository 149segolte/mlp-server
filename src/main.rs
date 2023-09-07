use axum::{routing::get, routing::post, Router};

#[tokio::main]
async fn main() {
    // build our application with a single route
    let app = Router::new()
        .route("/", get(projects))
        .route("/new", post(new_project));

    // run it with hyper on localhost:3000
    println!("Server running on localhost:3000");
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn projects() {}
async fn new_project() {}
