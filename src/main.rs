use anyhow::{Error as E, Result};
use axum::{extract::State, http::StatusCode, response::Json, routing::post, Router};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};
use std::{io::Write, sync::Arc};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    request_tx: mpsc::UnboundedSender<InferenceRequest>,
}

struct InferenceRequest {
    prompt: String,
    response_tx: oneshot::Sender<Result<String, String>>,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    response: String,
}

async fn model_worker(
    mut model: ModelForCausalLM,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    mut request_rx: mpsc::UnboundedReceiver<InferenceRequest>,
) {
    println!("🧠 Model worker started");
    // KV cache management could be added here in the future
    
    while let Some(request) = request_rx.recv().await {
        println!("🔍 Processing request: '{}'", request.prompt);
        
        let result = process_inference(&mut model, &tokenizer, &device, &request.prompt).await;
        
        // Send response back (ignore if receiver dropped)
        let _ = request.response_tx.send(result);
    }
    
    println!("📵 Model worker shutting down");
}

async fn process_inference(
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
) -> Result<String, String> {
    const MAX_TOKENS: usize = 50;
    const EOS_TOKEN: u32 = 151643; // Qwen2 EOS token
    
    // Clear KV cache ONCE at start of request
    model.clear_kv_cache();
    
    // Tokenize initial prompt
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("Tokenization failed: {}", e))?
        .get_ids()
        .to_vec();
    
    let mut generated_tokens = Vec::new();
    
    println!("🎯 Generating {} tokens for prompt: '{}'", MAX_TOKENS, prompt);
    
    // Generate tokens iteratively
    for step in 0..MAX_TOKENS {
        // For the first step, use full prompt. For subsequent steps, use only the last token
        let input_tokens = if step == 0 {
            tokens.clone()
        } else {
            vec![tokens[tokens.len() - 1]] // Only the last token
        };
        
        // Create input tensor
        let input_tensor = Tensor::new(input_tokens.as_slice(), device)
            .map_err(|e| format!("Tensor creation failed: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Unsqueeze failed: {}", e))?;
        
        // Forward pass with position offset for KV cache
        let start_pos = if step == 0 { 0 } else { tokens.len() - 1 };
        let logits = model
            .forward(&input_tensor, start_pos)
            .map_err(|e| format!("Forward pass failed at step {}: {}", step, e))?;
        
        // Extract last token logits
        let last_token_logits = logits
            .i((0, logits.dim(1).unwrap_or(1) - 1, ..))
            .map_err(|e| format!("Logits extraction failed: {}", e))?;
        
        let logits_vec = last_token_logits
            .to_vec1::<f32>()
            .map_err(|e| format!("Vector conversion failed: {}", e))?;
        
        // Find token with highest probability (greedy decoding)
        let next_token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        
        // Check for EOS token
        if next_token == EOS_TOKEN {
            println!("🛑 Hit EOS token at step {}", step);
            break;
        }
        
        // Add to generated tokens
        generated_tokens.push(next_token);
        tokens.push(next_token);
        
        // Decode and print progress
        if let Ok(token_text) = tokenizer.decode(&[next_token], false) {
            print!("{}", token_text);
            std::io::stdout().flush().ok();
        }
    }
    
    println!(); // New line after generation
    
    // Decode only the generated part (excluding original prompt)
    let response = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| format!("Decoding failed: {}", e))?;
    
    println!("✓ Generated {} tokens: '{}'", generated_tokens.len(), response);
    Ok(response)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Loading LLM for HTTP serving...");

    let device = Device::Cpu;
    
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2-0.5B".to_string());
    
    let tokenizer_filename = repo.get("tokenizer.json").await?;
    let config_filename = repo.get("config.json").await?;
    let model_filename = repo.get("model.safetensors").await?;
    
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config: Config = serde_json::from_value(config)?;
    
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
    let model = ModelForCausalLM::new(&config, vb)?;
    
    println!("✓ Model loaded successfully");

    // Create channel for model worker
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    
    // Spawn model worker task
    let tokenizer_arc = Arc::new(tokenizer);
    tokio::spawn(model_worker(model, tokenizer_arc.clone(), device, request_rx));

    let app_state = AppState { request_tx };

    let app = Router::new()
        .route("/generate", post(generate))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(app_state);

    println!("🚀 Starting HTTP server on http://0.0.0.0:3000");
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, StatusCode> {
    
    println!("📥 HTTP request: '{}'", request.prompt);
    
    // Create response channel
    let (response_tx, response_rx) = oneshot::channel();
    
    // Create inference request
    let inference_request = InferenceRequest {
        prompt: request.prompt,
        response_tx,
    };
    
    // Send to model worker
    if let Err(_) = state.request_tx.send(inference_request) {
        println!("❌ Failed to send request to model worker");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    
    // Wait for response
    match response_rx.await {
        Ok(Ok(response)) => {
            println!("✅ HTTP response: '{}'", response);
            Ok(Json(GenerateResponse { response }))
        },
        Ok(Err(error)) => {
            println!("❌ Model error: {}", error);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        },
        Err(_) => {
            println!("❌ Response channel closed");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
