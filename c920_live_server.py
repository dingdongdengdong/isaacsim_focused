#!/usr/bin/env python3
"""Browser-to-GPU live crack/rust segmentation service."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from aiohttp import WSMsgType, web
from ultralytics import YOLO


COLORS = {
    0: (0, 0, 255),  # crack: red (BGR)
    1: (255, 0, 0),  # rust: blue (BGR)
}

INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>C920 균열·부식 실시간 탐지</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#151c30; --line:#2b3655; --text:#e8ecf7; --muted:#9ca8c4; --accent:#54d6a5; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; font:15px/1.45 system-ui,sans-serif; background:radial-gradient(circle at top,#182442,var(--bg) 50%); color:var(--text); }
    main { width:min(1180px,calc(100% - 28px)); margin:24px auto; }
    h1 { margin:0 0 4px; font-size:clamp(22px,3vw,34px); }
    .sub { color:var(--muted); margin-bottom:18px; }
    .panel { background:rgba(21,28,48,.94); border:1px solid var(--line); border-radius:16px; box-shadow:0 18px 55px #0006; }
    .controls { display:grid; grid-template-columns:minmax(220px,1fr) auto minmax(210px,.7fr) auto auto; gap:10px; align-items:end; padding:14px; }
    label { display:grid; gap:5px; color:var(--muted); font-size:13px; }
    select,button,input { font:inherit; }
    select,button { min-height:40px; border:1px solid var(--line); border-radius:9px; background:#0f172a; color:var(--text); padding:8px 12px; }
    button { cursor:pointer; font-weight:700; }
    button.primary { background:var(--accent); color:#06261b; border-color:transparent; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    input[type=range] { width:100%; accent-color:var(--accent); }
    .stage { position:relative; overflow:hidden; aspect-ratio:16/9; background:#040710; border-radius:16px 16px 0 0; }
    #result { width:100%; height:100%; object-fit:contain; display:block; }
    #source { display:none; }
    .badge { position:absolute; top:12px; left:12px; padding:6px 10px; border-radius:99px; background:#000b; color:#ff8792; font-weight:800; }
    .badge.live { color:#69e6b5; }
    .stats { display:grid; grid-template-columns:repeat(5,1fr); gap:1px; background:var(--line); border-top:1px solid var(--line); }
    .stat { background:var(--panel); padding:12px 14px; }
    .stat b { display:block; font-size:20px; margin-top:2px; }
    .detail { padding:13px 15px; color:var(--muted); min-height:48px; border-top:1px solid var(--line); }
    .legend { display:flex; gap:18px; padding:12px 15px; border-top:1px solid var(--line); color:var(--muted); }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
    #error { color:#ff99a2; white-space:pre-wrap; }
    @media (max-width:800px) { .controls { grid-template-columns:1fr 1fr; } .controls label:first-child { grid-column:1/-1; } .stats { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body><main>
  <h1>C920 균열·부식 실시간 탐지</h1>
  <div class="sub">로컬 카메라 → SSH 터널 → RTX 5070 YOLO 세그멘테이션</div>
  <section class="panel controls">
    <label>카메라<select id="camera"></select></label>
    <button id="refresh">목록 새로고침</button>
    <label>신뢰도 <span id="confText">0.15</span><input id="confidence" type="range" min="0.05" max="0.90" value="0.15" step="0.05"></label>
    <button id="start" class="primary">시작</button>
    <button id="stop" disabled>중지</button>
    <button id="snapshot" disabled>대표 장면 저장</button>
  </section>
  <section class="panel" style="margin-top:14px">
    <div class="stage"><video id="source" autoplay playsinline muted></video><img id="result" alt="추론 결과"><span id="status" class="badge">대기 중</span></div>
    <div class="stats">
      <div class="stat">탐지 수<b id="count">—</b></div>
      <div class="stat">실시간 FPS<b id="fps">—</b></div>
      <div class="stat">추론 시간<b id="infer">—</b></div>
      <div class="stat">왕복 지연<b id="latency">—</b></div>
      <div class="stat">프레임 크기<b id="size">—</b></div>
    </div>
    <div id="detail" class="detail">시작을 누르고 브라우저의 카메라 권한을 허용하세요.</div>
    <div class="legend"><span><i class="dot" style="background:#f22"></i>균열</span><span><i class="dot" style="background:#2587ff"></i>부식</span><span id="saved"></span></div>
    <div id="error" class="detail" hidden></div>
  </section>
  <canvas id="capture" hidden></canvas>
</main>
<script>
const $ = id => document.getElementById(id);
let stream=null, ws=null, running=false, waiting=false, sentAt=0, lastResultAt=0, smoothFps=0, resultUrl=null;

async function enumerateCameras(requestPermission=false) {
  if (requestPermission && !stream) {
    const probe=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    probe.getTracks().forEach(t=>t.stop());
  }
  const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
  const old=$('camera').value; $('camera').innerHTML='';
  devices.forEach((d,i)=>{ const o=document.createElement('option'); o.value=d.deviceId; o.textContent=d.label||`카메라 ${i+1}`; $('camera').appendChild(o); });
  if ([...$('camera').options].some(o=>o.value===old)) $('camera').value=old;
  const c920=[...$('camera').options].find(o=>/c920/i.test(o.textContent)); if(c920) $('camera').value=c920.value;
}

function setError(err) { $('error').hidden=!err; $('error').textContent=err ? String(err?.message||err) : ''; }
function setRunning(on) { running=on; $('start').disabled=on; $('stop').disabled=!on; $('snapshot').disabled=!on; $('status').textContent=on?'LIVE':'대기 중'; $('status').classList.toggle('live',on); }
function stopAll() {
  setRunning(false); waiting=false;
  if(ws){ const old=ws; ws=null; old.close(); }
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  $('source').srcObject=null;
}

function connectSocket() {
  return new Promise((resolve,reject)=>{
    const proto=location.protocol==='https:'?'wss':'ws'; ws=new WebSocket(`${proto}://${location.host}/ws`); ws.binaryType='arraybuffer';
    ws.onopen=()=>{ ws.send(JSON.stringify({type:'config',confidence:Number($('confidence').value)})); resolve(); };
    ws.onerror=()=>reject(new Error('서버 WebSocket 연결에 실패했습니다.'));
    ws.onclose=()=>{ if(running){ setError('서버 연결이 종료되었습니다.'); stopAll(); } };
    ws.onmessage=handleResult;
  });
}

function handleResult(event) {
  if(typeof event.data==='string') {
    try { const message=JSON.parse(event.data); if(message.type==='error') throw new Error(message.message); } catch(e) { setError(e); stopAll(); }
    return;
  }
  if(!(event.data instanceof ArrayBuffer)) return;
  const buf=event.data, view=new DataView(buf), n=view.getUint32(0), meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,4,n)));
  const jpeg=new Blob([buf.slice(4+n)],{type:'image/jpeg'});
  if(resultUrl) URL.revokeObjectURL(resultUrl); resultUrl=URL.createObjectURL(jpeg); $('result').src=resultUrl;
  const now=performance.now(), dt=lastResultAt?1000/(now-lastResultAt):0; lastResultAt=now; smoothFps=smoothFps?0.8*smoothFps+0.2*dt:dt;
  $('count').textContent=meta.count; $('fps').textContent=smoothFps?smoothFps.toFixed(1):'—'; $('infer').textContent=`${meta.inference_ms.toFixed(0)} ms`; $('latency').textContent=`${(now-sentAt).toFixed(0)} ms`; $('size').textContent=`${meta.width}×${meta.height}`;
  $('detail').textContent=meta.detections.length ? meta.detections.map(d=>`${d.name} ${(d.confidence*100).toFixed(1)}%`).join(' · ') : '탐지 없음';
  if(meta.saved) $('saved').textContent=`저장됨: ${meta.saved}`;
  waiting=false; if(running) requestAnimationFrame(sendFrame);
}

function sendFrame() {
  if(!running || waiting || !ws || ws.readyState!==WebSocket.OPEN || $('source').readyState<2) return;
  const v=$('source'), maxW=960, scale=Math.min(1,maxW/v.videoWidth), c=$('capture'); c.width=Math.round(v.videoWidth*scale); c.height=Math.round(v.videoHeight*scale);
  c.getContext('2d').drawImage(v,0,0,c.width,c.height); waiting=true; sentAt=performance.now();
  c.toBlob(async blob=>{ if(!blob){ waiting=false; return; } try{ ws.send(await blob.arrayBuffer()); }catch(e){ setError(e); stopAll(); } },'image/jpeg',0.82);
}

async function start() {
  setError(''); try {
    await enumerateCameras(true);
    const deviceId=$('camera').value; stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:deviceId?{exact:deviceId}:undefined,width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false});
    $('source').srcObject=stream; await $('source').play(); await connectSocket(); setRunning(true); requestAnimationFrame(sendFrame);
  } catch(e) { setError(e); stopAll(); }
}

$('start').onclick=start; $('stop').onclick=stopAll;
$('refresh').onclick=()=>enumerateCameras(true).catch(setError);
$('confidence').oninput=()=>{ $('confText').textContent=Number($('confidence').value).toFixed(2); if(ws?.readyState===1) ws.send(JSON.stringify({type:'config',confidence:Number($('confidence').value)})); };
$('snapshot').onclick=()=>{ if(ws?.readyState===1){ ws.send(JSON.stringify({type:'snapshot'})); $('saved').textContent='다음 장면 저장 요청됨…'; } };
navigator.mediaDevices?.addEventListener?.('devicechange',()=>enumerateCameras(false)); enumerateCameras(false).catch(setError);
</script></body></html>"""


@dataclass
class InferenceOutput:
    original: np.ndarray
    rendered: np.ndarray
    jpeg: bytes
    metadata: dict[str, Any]


class InferenceEngine:
    def __init__(self, model_path: Path, run_dir: Path, device: str, imgsz: int) -> None:
        self.model = YOLO(str(model_path))
        self.run_dir = run_dir
        self.device = device
        self.imgsz = imgsz
        self.lock = Lock()
        self.frame_index = 0
        self.auto_saved_classes: set[int] = set()
        self.metrics_path = run_dir / "metrics.jsonl"

    def infer(self, payload: bytes, confidence: float, force_snapshot: bool) -> InferenceOutput:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG frame decode failed")
        started = time.perf_counter()
        with self.lock:
            result = self.model.predict(source=image, imgsz=self.imgsz, conf=confidence, device=self.device, verbose=False)[0]
        inference_ms = (time.perf_counter() - started) * 1000
        rendered = image.copy()
        detections: list[dict[str, Any]] = []
        seen_classes: set[int] = set()

        if result.masks is not None and result.boxes is not None:
            for mask_tensor, box in zip(result.masks.data, result.boxes):
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                name = str(result.names[class_id])
                color = COLORS.get(class_id, (0, 255, 255))
                mask = cv2.resize(mask_tensor.cpu().numpy(), (image.shape[1], image.shape[0])) > 0.5
                colored = np.full_like(rendered, color)
                rendered[mask] = cv2.addWeighted(rendered, 0.35, colored, 0.65, 0)[mask]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rendered, contours, -1, color, 2)
                x1, y1, _, _ = (int(v) for v in box.xyxy[0].tolist())
                label = f"{name} {score:.2f}"
                cv2.putText(rendered, label, (x1, max(y1 - 7, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                detections.append({"class_id": class_id, "name": name, "confidence": score})
                seen_classes.add(class_id)

        self.frame_index += 1
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        saved: str | None = None
        new_classes = seen_classes - self.auto_saved_classes
        if force_snapshot or new_classes:
            stem = f"frame-{self.frame_index:06d}"
            cv2.imwrite(str(self.run_dir / f"{stem}-original.jpg"), image)
            cv2.imwrite(str(self.run_dir / f"{stem}-result.jpg"), rendered)
            saved = stem
            self.auto_saved_classes.update(seen_classes)

        metadata: dict[str, Any] = {
            "timestamp": timestamp,
            "frame": self.frame_index,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "confidence_threshold": confidence,
            "inference_ms": inference_ms,
            "count": len(detections),
            "detections": detections,
            "saved": saved,
        }
        with self.metrics_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        ok, jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Annotated JPEG encode failed")
        return InferenceOutput(image, rendered, jpg.tobytes(), metadata)


async def index(_: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def health(request: web.Request) -> web.Response:
    engine: InferenceEngine = request.app["engine"]
    return web.json_response({
        "ok": True,
        "model": str(request.app["model_path"]),
        "device": engine.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_dir": str(engine.run_dir),
    })


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=12 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    confidence = 0.15
    snapshot_next = False
    engine: InferenceEngine = request.app["engine"]
    async for msg in ws:
        try:
            if msg.type == WSMsgType.TEXT:
                command = json.loads(msg.data)
                if command.get("type") == "config":
                    confidence = min(0.90, max(0.05, float(command.get("confidence", confidence))))
                elif command.get("type") == "snapshot":
                    snapshot_next = True
            elif msg.type == WSMsgType.BINARY:
                output = await asyncio.to_thread(engine.infer, msg.data, confidence, snapshot_next)
                snapshot_next = False
                meta = json.dumps(output.metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                await ws.send_bytes(struct.pack(">I", len(meta)) + meta + output.jpeg)
            elif msg.type == WSMsgType.ERROR:
                break
        except Exception as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
    return ws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("runs/dacl10k-crack-rust-yolo26n-seg-30e-b24/weights/best.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    session = datetime.now().astimezone().strftime("c920-live-%Y%m%d-%H%M%S")
    run_dir = (Path("runs") / session).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    engine = InferenceEngine(model_path, run_dir, args.device, args.imgsz)
    app = web.Application(client_max_size=12 * 1024 * 1024)
    app["engine"] = engine
    app["model_path"] = model_path
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket_handler)
    print(json.dumps({"event": "ready", "url": f"http://{args.host}:{args.port}", "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
