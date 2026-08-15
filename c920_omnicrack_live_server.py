#!/usr/bin/env python3
"""C920 browser stream backed by the official OmniCrack30k nnU-Net ensemble."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from aiohttp import WSMsgType, web
from skimage.morphology import thin


INDEX_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCrack30k C920 Live Crack Segmentation</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#151c30;--line:#2b3655;--text:#e8ecf7;--muted:#9ca8c4;--accent:#54d6a5}*{box-sizing:border-box}
body{margin:0;min-height:100vh;font:15px/1.45 system-ui,sans-serif;background:radial-gradient(circle at top,#182442,var(--bg) 50%);color:var(--text)}
main{width:min(1180px,calc(100% - 28px));margin:24px auto}h1{margin:0 0 4px;font-size:clamp(22px,3vw,34px)}.sub{color:var(--muted);margin-bottom:18px}
.panel{background:rgba(21,28,48,.94);border:1px solid var(--line);border-radius:16px;box-shadow:0 18px 55px #0006}.controls{display:grid;grid-template-columns:minmax(220px,1fr) auto minmax(180px,.7fr) auto auto auto;gap:10px;align-items:end;padding:14px}
label{display:grid;gap:5px;color:var(--muted);font-size:13px}select,button,input{font:inherit}select,button{min-height:40px;border:1px solid var(--line);border-radius:9px;background:#0f172a;color:var(--text);padding:8px 12px}button{cursor:pointer;font-weight:700}button.primary{background:var(--accent);color:#06261b;border-color:transparent}button:disabled{opacity:.45;cursor:not-allowed}input[type=range]{width:100%;accent-color:var(--accent)}
.stage{position:relative;overflow:hidden;aspect-ratio:16/9;background:#040710;border-radius:16px 16px 0 0}#result{width:100%;height:100%;object-fit:contain;display:block}#source{display:none}.badge{position:absolute;top:12px;left:12px;padding:6px 10px;border-radius:99px;background:#000b;color:#ff8792;font-weight:800}.badge.live{color:#69e6b5}
.stats{display:grid;grid-template-columns:repeat(7,1fr);gap:1px;background:var(--line);border-top:1px solid var(--line)}.stat{background:var(--panel);padding:12px 14px}.stat b{display:block;font-size:19px;margin-top:2px}.detail,.legend{padding:13px 15px;color:var(--muted);border-top:1px solid var(--line)}.legend{display:flex;gap:18px;flex-wrap:wrap}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}#error{color:#ff99a2;white-space:pre-wrap}
@media(max-width:850px){.controls{grid-template-columns:1fr 1fr}.controls label:first-child{grid-column:1/-1}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<h1>OmniCrack30k · C920 Live</h1><div class="sub">Official 4-fold nnU-Net ensemble · binary structural crack segmentation · RTX 5070</div>
<section class="panel controls">
<label>Camera<select id="camera"></select></label><button id="refresh">Refresh cameras</button>
<label>Crack threshold <span id="confText">0.50</span><input id="confidence" type="range" min="0.05" max="0.95" value="0.50" step="0.05"></label>
<label>Input width<select id="resolution"><option value="512">512 · faster</option><option value="640" selected>640 · balanced</option><option value="960">960 · detail</option></select></label>
<button id="start" class="primary">Start</button><button id="stop" disabled>Stop</button><button id="snapshot" disabled>Save frame</button>
</section>
<section class="panel" style="margin-top:14px"><div class="stage"><video id="source" autoplay playsinline muted></video><img id="result" alt="OmniCrack result"><span id="status" class="badge">READY</span></div>
<div class="stats"><div class="stat">Crack regions ≥8px<b id="count">—</b></div><div class="stat">Crack area<b id="area">—</b></div><div class="stat">Centerline<b id="centerline">—</b></div><div class="stat">Display FPS<b id="fps">—</b></div><div class="stat">GPU inference<b id="infer">—</b></div><div class="stat">Round trip<b id="latency">—</b></div><div class="stat">Frame size<b id="size">—</b></div></div>
<div id="detail" class="detail">Allow camera permission, select HD Pro Webcam C920, then press Start.</div>
<div class="legend"><span><i class="dot" style="background:#f22"></i>Crack mask</span><span><i class="dot" style="background:#ffd21f"></i>One-pixel centerline</span><span>Not corrosion detection · no physical units without calibration</span><span id="saved"></span></div><div id="error" class="detail" hidden></div></section><canvas id="capture" hidden></canvas>
</main><script>
const $=id=>document.getElementById(id);let stream=null,ws=null,running=false,waiting=false,sentAt=0,lastResultAt=0,smoothFps=0,resultUrl=null;
async function enumerateCameras(requestPermission=false){if(requestPermission&&!stream){const p=await navigator.mediaDevices.getUserMedia({video:true,audio:false});p.getTracks().forEach(t=>t.stop())}const ds=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput'),old=$('camera').value;$('camera').innerHTML='';ds.forEach((d,i)=>{const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||`Camera ${i+1}`;$('camera').appendChild(o)});if([...$('camera').options].some(o=>o.value===old))$('camera').value=old;const c920=[...$('camera').options].find(o=>/c920/i.test(o.textContent));if(c920)$('camera').value=c920.value}
function setError(e){$('error').hidden=!e;$('error').textContent=e?String(e?.message||e):''}function setRunning(on){running=on;$('start').disabled=on;$('stop').disabled=!on;$('snapshot').disabled=!on;$('status').textContent=on?'LIVE':'READY';$('status').classList.toggle('live',on)}
function stopAll(){setRunning(false);waiting=false;if(ws){const old=ws;ws=null;old.close()}if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}$('source').srcObject=null}
function connectSocket(){return new Promise((resolve,reject)=>{const proto=location.protocol==='https:'?'wss':'ws';ws=new WebSocket(`${proto}://${location.host}/ws`);ws.binaryType='arraybuffer';ws.onopen=()=>{ws.send(JSON.stringify({type:'config',confidence:Number($('confidence').value)}));resolve()};ws.onerror=()=>reject(new Error('WebSocket connection failed.'));ws.onclose=()=>{if(running){setError('Server connection closed.');stopAll()}};ws.onmessage=handleResult})}
function handleResult(event){if(typeof event.data==='string'){try{const m=JSON.parse(event.data);if(m.type==='error')throw new Error(m.message)}catch(e){setError(e);stopAll()}return}if(!(event.data instanceof ArrayBuffer))return;const buf=event.data,n=new DataView(buf).getUint32(0),meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,4,n))),jpeg=new Blob([buf.slice(4+n)],{type:'image/jpeg'});if(resultUrl)URL.revokeObjectURL(resultUrl);resultUrl=URL.createObjectURL(jpeg);$('result').src=resultUrl;const now=performance.now(),rtt=now-sentAt,instant=lastResultAt?1000/(now-lastResultAt):0;lastResultAt=now;smoothFps=smoothFps?0.8*smoothFps+0.2*instant:instant;$('count').textContent=meta.region_count;$('area').textContent=`${meta.crack_fraction_pct.toFixed(2)}%`;$('centerline').textContent=`${meta.centerline_pixels}px`;$('fps').textContent=smoothFps?smoothFps.toFixed(2):'—';$('infer').textContent=`${meta.inference_ms.toFixed(0)}ms`;$('latency').textContent=`${rtt.toFixed(0)}ms`;$('size').textContent=`${meta.width}×${meta.height}`;$('detail').textContent=meta.crack_pixels?`Crack pixels ${meta.crack_pixels.toLocaleString()} · max probability ${(meta.max_probability*100).toFixed(1)}% · mean crack probability ${(meta.mean_crack_probability*100).toFixed(1)}%`:'No crack pixels above the selected threshold.';if(meta.saved)$('saved').textContent=`Saved: ${meta.saved}`;if(ws?.readyState===1)ws.send(JSON.stringify({type:'client_metrics',frame:meta.frame,roundtrip_ms:rtt,display_fps:smoothFps}));waiting=false;if(running)requestAnimationFrame(sendFrame)}
function sendFrame(){if(!running||waiting||!ws||ws.readyState!==WebSocket.OPEN||$('source').readyState<2)return;const v=$('source'),maxW=Number($('resolution').value),scale=Math.min(1,maxW/v.videoWidth),c=$('capture');c.width=Math.round(v.videoWidth*scale);c.height=Math.round(v.videoHeight*scale);c.getContext('2d').drawImage(v,0,0,c.width,c.height);waiting=true;sentAt=performance.now();c.toBlob(async b=>{if(!b){waiting=false;return}try{ws.send(await b.arrayBuffer())}catch(e){setError(e);stopAll()}},'image/jpeg',0.88)}
async function start(){setError('');try{await enumerateCameras(true);const id=$('camera').value;stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:id?{exact:id}:undefined,width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false});$('source').srcObject=stream;await $('source').play();await connectSocket();setRunning(true);requestAnimationFrame(sendFrame)}catch(e){setError(e);stopAll()}}
$('start').onclick=start;$('stop').onclick=stopAll;$('refresh').onclick=()=>enumerateCameras(true).catch(setError);$('confidence').oninput=()=>{$('confText').textContent=Number($('confidence').value).toFixed(2);if(ws?.readyState===1)ws.send(JSON.stringify({type:'config',confidence:Number($('confidence').value)}))};$('snapshot').onclick=()=>{if(ws?.readyState===1){ws.send(JSON.stringify({type:'snapshot'}));$('saved').textContent='Save requested…'}};navigator.mediaDevices?.addEventListener?.('devicechange',()=>enumerateCameras(false));enumerateCameras(false).catch(setError);
</script></body></html>"""


class InferenceEngine:
    def __init__(self, model_root: Path, run_dir: Path, folds: tuple[int, ...]) -> None:
        sys.path.insert(0, str(model_root))
        from omnicrack30k.inference import OmniCrack30kModel

        self.model = OmniCrack30kModel(folds=folds, allow_tqdm=False)
        self.run_dir = run_dir
        self.folds = folds
        self.lock = Lock()
        self.log_lock = Lock()
        self.frame_index = 0
        self.auto_saved_crack = False
        self.metrics_path = run_dir / "metrics.jsonl"
        self.client_metrics_path = run_dir / "client_metrics.jsonl"

    def infer(self, payload: bytes, threshold: float, force_snapshot: bool) -> tuple[bytes, dict[str, Any]]:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG frame decode failed")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with self.lock:
            softmax, _ = self.model(rgb, rgb=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        probability = np.asarray(softmax[1], dtype=np.float32)
        prediction = probability >= threshold
        centerline = thin(prediction)
        crack_pixels = int(prediction.sum())
        centerline_pixels = int(centerline.sum())
        crack_fraction_pct = float(prediction.mean() * 100)
        max_probability = float(probability.max())
        mean_crack_probability = float(probability[prediction].mean()) if crack_pixels else 0.0
        components, _, stats, _ = cv2.connectedComponentsWithStats(prediction.astype(np.uint8), 8)
        region_count = sum(int(stats[i, cv2.CC_STAT_AREA]) >= 8 for i in range(1, components))

        rendered = image.copy()
        red = np.full_like(rendered, (0, 0, 255))
        rendered[prediction] = cv2.addWeighted(rendered, 0.35, red, 0.65, 0)[prediction]
        rendered[centerline] = (0, 215, 255)
        contours, _ = cv2.findContours(prediction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, (0, 0, 255), 1)
        label = f"crack area {crack_fraction_pct:.2f}% | regions {region_count} | threshold {threshold:.2f}"
        cv2.rectangle(rendered, (0, 0), (min(rendered.shape[1], 650), 34), (0, 0, 0), -1)
        cv2.putText(rendered, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        self.frame_index += 1
        saved: str | None = None
        if force_snapshot or (crack_pixels and not self.auto_saved_crack):
            stem = f"frame-{self.frame_index:06d}"
            cv2.imwrite(str(self.run_dir / f"{stem}-original.jpg"), image)
            cv2.imwrite(str(self.run_dir / f"{stem}-result.jpg"), rendered)
            cv2.imwrite(str(self.run_dir / f"{stem}-probability.png"), np.uint8(np.clip(probability, 0, 1) * 255))
            cv2.imwrite(str(self.run_dir / f"{stem}-mask.png"), np.uint8(prediction) * 255)
            cv2.imwrite(str(self.run_dir / f"{stem}-centerline.png"), np.uint8(centerline) * 255)
            saved = stem
            self.auto_saved_crack = self.auto_saved_crack or bool(crack_pixels)

        metadata: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "frame": self.frame_index,
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "threshold": threshold, "inference_ms": inference_ms,
            "crack_pixels": crack_pixels, "crack_fraction_pct": crack_fraction_pct,
            "centerline_pixels": centerline_pixels, "region_count": region_count,
            "max_probability": max_probability, "mean_crack_probability": mean_crack_probability,
            "saved": saved,
        }
        with self.log_lock, self.metrics_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        ok, jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Annotated JPEG encode failed")
        return jpg.tobytes(), metadata

    def record_client_metrics(self, values: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "frame": int(values.get("frame", 0)),
            "roundtrip_ms": float(values.get("roundtrip_ms", 0)),
            "display_fps": float(values.get("display_fps", 0)),
        }
        with self.log_lock, self.client_metrics_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")


async def index(_: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def health(request: web.Request) -> web.Response:
    engine: InferenceEngine = request.app["engine"]
    return web.json_response({"ok": True, "model": "official OmniCrack30k nnU-Net 2D", "folds": engine.folds, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "run_dir": str(engine.run_dir)})


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=12 * 1024 * 1024, heartbeat=30)
    await ws.prepare(request)
    threshold, snapshot_next = 0.5, False
    engine: InferenceEngine = request.app["engine"]
    async for msg in ws:
        try:
            if msg.type == WSMsgType.TEXT:
                command = json.loads(msg.data)
                if command.get("type") == "config":
                    threshold = min(0.95, max(0.05, float(command.get("confidence", threshold))))
                elif command.get("type") == "snapshot":
                    snapshot_next = True
                elif command.get("type") == "client_metrics":
                    engine.record_client_metrics(command)
            elif msg.type == WSMsgType.BINARY:
                jpeg, metadata = await asyncio.to_thread(engine.infer, msg.data, threshold, snapshot_next)
                snapshot_next = False
                meta = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                await ws.send_bytes(struct.pack(">I", len(meta)) + meta + jpeg)
            elif msg.type == WSMsgType.ERROR:
                break
        except Exception as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
    return ws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/home/dong/ai/external/omnicrack30k/src"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    run_dir = (Path("runs") / datetime.now().astimezone().strftime("c920-omnicrack-live-%Y%m%d-%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    engine = InferenceEngine(args.model_root.resolve(), run_dir, tuple(args.folds))
    app = web.Application(client_max_size=12 * 1024 * 1024)
    app["engine"] = engine
    app.router.add_get("/", index); app.router.add_get("/health", health); app.router.add_get("/ws", websocket_handler)
    print(json.dumps({"event": "ready", "url": f"http://{args.host}:{args.port}", "run_dir": str(run_dir), "folds": args.folds}), flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
