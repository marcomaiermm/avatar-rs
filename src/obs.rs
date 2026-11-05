//! Minimal OBS WebSocket v5 client: just enough to switch scenes and toggle a source.
//! One background thread owns the connection; it connects lazily on the first action.

use crate::config::{Action, Obs};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::Duration;
use tungstenite::{Message, WebSocket, stream::MaybeTlsStream};

type Ws = WebSocket<MaybeTlsStream<TcpStream>>;

pub fn spawn(cfg: Obs) -> mpsc::Sender<Action> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut ws: Option<Ws> = None;
        let mut reported = false;
        for action in rx {
            // A cached connection may have died since the last action: retry once on a fresh one.
            for _ in 0..2 {
                let fresh = ws.is_none();
                if fresh {
                    match connect(&cfg) {
                        Ok(c) => (ws, reported) = (Some(c), false),
                        Err(e) => {
                            if !std::mem::replace(&mut reported, true) {
                                eprintln!("obs: {e}");
                            }
                            break;
                        }
                    }
                }
                match run(ws.as_mut().unwrap(), &action) {
                    Ok(()) => break,
                    Err(e) => {
                        ws = None;
                        if fresh {
                            eprintln!("obs: {e}");
                            break;
                        }
                    }
                }
            }
        }
    });
    tx
}

fn auth(password: &str, challenge: &str, salt: &str) -> String {
    let sha = |s: String| B64.encode(Sha256::digest(s));
    sha(sha(format!("{password}{salt}")) + challenge)
}

fn connect(cfg: &Obs) -> Result<Ws, String> {
    let (mut ws, _) = tungstenite::connect(cfg.url.as_str()).map_err(|e| format!("connect {}: {e}", cfg.url))?;
    if let MaybeTlsStream::Plain(s) = ws.get_mut() {
        s.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    }
    let hello = read_op(&mut ws, 0)?;
    let mut identify = json!({ "rpcVersion": 1, "eventSubscriptions": 0 });
    if let Some(a) = hello["d"].get("authentication") {
        let field = |k: &str| a[k].as_str().unwrap_or_default();
        identify["authentication"] = auth(&cfg.password, field("challenge"), field("salt")).into();
    }
    send(&mut ws, 1, identify)?;
    read_op(&mut ws, 2).map_err(|e| format!("identify (wrong password?): {e}"))?;
    Ok(ws)
}

fn send(ws: &mut Ws, op: u8, d: Value) -> Result<(), String> {
    ws.send(Message::text(json!({ "op": op, "d": d }).to_string())).map_err(|e| e.to_string())
}

/// Reads until a message with this opcode arrives.
fn read_op(ws: &mut Ws, op: u8) -> Result<Value, String> {
    loop {
        match ws.read().map_err(|e| e.to_string())? {
            Message::Text(t) => {
                let v: Value = serde_json::from_str(&t).map_err(|e| e.to_string())?;
                if v["op"] == op {
                    return Ok(v);
                }
            }
            Message::Close(_) => return Err("connection closed".into()),
            _ => {}
        }
    }
}

/// Returns `responseData`; a failed request status becomes an error.
fn request(ws: &mut Ws, ty: &str, data: Value) -> Result<Value, String> {
    // Requests are sequential and answers matched by id, so a fixed id per type is enough.
    send(ws, 6, json!({ "requestType": ty, "requestId": ty, "requestData": data }))?;
    loop {
        let v = read_op(ws, 7)?;
        if v["d"]["requestId"] != ty {
            continue;
        }
        let status = &v["d"]["requestStatus"];
        return if status["result"] == true {
            Ok(v["d"]["responseData"].clone())
        } else {
            Err(format!("{ty}: {}", status["comment"].as_str().unwrap_or("request failed")))
        };
    }
}

fn run(ws: &mut Ws, action: &Action) -> Result<(), String> {
    match action {
        Action::ObsScene(scene) => {
            request(ws, "SetCurrentProgramScene", json!({ "sceneName": scene }))?;
        }
        Action::ObsToggleSource { scene, source } => {
            let id = request(ws, "GetSceneItemId", json!({ "sceneName": scene, "sourceName": source }))?["sceneItemId"].clone();
            let mut item = json!({ "sceneName": scene, "sceneItemId": id });
            let enabled = request(ws, "GetSceneItemEnabled", item.clone())?["sceneItemEnabled"].as_bool().unwrap_or(false);
            item["sceneItemEnabled"] = (!enabled).into();
            request(ws, "SetSceneItemEnabled", item)?;
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn auth_string() {
        // Expected value computed independently with python3 hashlib/base64.
        assert_eq!(super::auth("supersecretpassword", "challenge456", "salt123"), "6twMlPBVj3ybvQB4m04DkbGv/NVjl7ed9emm/qfGi0g=");
    }
}
