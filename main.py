from datetime import timedelta, datetime
import threading
import time
import os
import traceback
from typing import Optional, Set
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from db import get_cameras, get_camera_rtsp, save_ocorrencia, save_video_record
from ffmpeg_service import check_hls_activity, save_hls_clip, start_stream, stop_stream, save_hls_as_mp4_fast, STREAMS_DIR, last_seen
from analysis_manager import start_analysis_for_camera, stop_analysis_for_camera, class_counters
import torch
import ipaddress
import socket
import concurrent.futures



FOTOS_DIR = "fotos"
os.makedirs(FOTOS_DIR, exist_ok=True)

load_dotenv()

app = FastAPI()
known_cameras: Set[str] = set()
camera_scan_event = threading.Event()

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:5173",  # Vite
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def restrict_access(request: Request, call_next):
    path = request.url.path
    blocked_patterns = [
        ".git", ".env", "__pycache__", "templates", "fotos", 
        "ffmpeg_service", "analysis_manager", ".vscode"
    ]
    
    if any(bp in path for bp in blocked_patterns):
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    # Se não bateu em nenhum bloqueio, deixa passar
    return await call_next(request)



app.mount("/streams", StaticFiles(directory=STREAMS_DIR), name="streams")

from google.cloud import storage
google_creds = {
    "type": "service_account",
    "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
    "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL"),
    "client_id": os.getenv("GOOGLE_CLOUD_CLIENT_ID"),
    "auth_uri": os.getenv("GOOGLE_CLOUD_AUTH_URI"),
    "token_uri": os.getenv("GOOGLE_CLOUD_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GOOGLE_CLOUD_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("GOOGLE_CLOUD_CLIENT_CERT_URL"),
    "universe_domain": os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN"),
}
client = storage.Client.from_service_account_info(google_creds)
bucket = client.bucket("acailandia")

def upload_video_to_gcs(local_path: str, bucket_name: str = "acailandia") -> str:
    filename = os.path.basename(local_path)
    blob = bucket.blob(f"cams/{filename}")
    blob.upload_from_filename(local_path)
    signed_url = blob.generate_signed_url(
        version="v2",
        expiration=timedelta(days=365*5),
        method="GET"
    )
    return signed_url

# Dicionários para controle
active_streams = {}
active_analyses = {}
save_events = {}
save_threads = {}

def clean_old_ts_files(camera_id: int, max_age_hours=6):
    """Limpa apenas arquivos .ts muito antigos"""
    try:
        ts_files = [
            f for f in os.listdir(STREAMS_DIR)
            if f.startswith(f"camera_{camera_id}_") and f.endswith(".ts")
        ]
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for ts_file in ts_files:
            ts_path = os.path.join(STREAMS_DIR, ts_file)
            if os.path.exists(ts_path):
                file_age = current_time - os.path.getmtime(ts_path)
                
                if file_age > max_age_seconds:
                    try:
                        os.remove(ts_path)
                        print(f"🗑️  Arquivo antigo removido: {ts_file}")
                    except Exception as e:
                        print(f"⚠️  Erro ao remover {ts_file}: {e}")
                        
    except Exception as e:
        print(f"❌ Erro ao limpar arquivos TS da câmera {camera_id}: {e}")

def clean_old_mp4_files(max_age_hours=48):
    """
    Remove arquivos MP4 antigos do diretório de streams (ou outro diretório definido).
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        # procura por mp4 em STREAMS_DIR e subpastas
        for root, _, files in os.walk(STREAMS_DIR):
            for f in files:
                if f.endswith(".mp4"):
                    file_path = os.path.join(root, f)
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            print(f"🗑️  MP4 antigo removido: {file_path}")
                        except Exception as e:
                            print(f"⚠️  Erro ao remover {file_path}: {e}")

    except Exception as e:
        print(f"❌ Erro ao limpar MP4s antigos: {e}")
        
def start_cleanup_scheduler(interval_hours=48):
    """Thread que executa limpezas periódicas de arquivos antigos"""
    def cleaner():
        while True:
            print(f"🧹 Executando limpeza programada de arquivos antigos...")
            clean_old_ts_files(camera_id=0)  # pode ajustar pra todas
            clean_old_mp4_files()
            print(f"✅ Limpeza concluída. Próxima em {interval_hours}h")
            time.sleep(interval_hours * 3600)

    threading.Thread(target=cleaner, daemon=True).start()



def schedule_auto_save(camera_id: int, interval_seconds=7200):
    """
    Salva um MP4 automaticamente a cada X segundos (para teste).
    Exemplo: schedule_auto_save(1, interval_seconds=30)
    """
    print(f"⏰ Iniciando auto-save para câmera {camera_id} a cada {interval_seconds} segundos")
    stop_event = threading.Event()
    save_events[camera_id] = stop_event

    def job():
        execution_count = 0

        while not stop_event.is_set():
            try:
                start_exec = time.time()  # início real
                execution_count += 1
                print(f"\n🔄 Execução #{execution_count} do auto-save para câmera {camera_id}")

                # Define o período apenas para log
                end_time = datetime.now()
                start_time = end_time - timedelta(seconds=interval_seconds)

                output_path = save_hls_as_mp4_fast(camera_id, last_seconds=interval_seconds, delete_ts=True)

                if output_path and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"✅ Vídeo salvo: {output_path} ({file_size:.2f} MB)")
                else:
                    print(f"⚠️ Nenhum vídeo foi gerado para câmera {camera_id}")

            except Exception as e:
                print(f"❌ Erro durante o processamento da câmera {camera_id}: {e}")
                import traceback
                traceback.print_exc()

            # tempo total de execução até aqui
            elapsed = time.time() - start_exec
            wait_time = max(0, interval_seconds - elapsed)  # evita valores negativos

            print(f"⏳ Aguardando {wait_time:.0f} segundos até a próxima gravação...")

            waited = 0
            while waited < wait_time and not stop_event.is_set():
                time.sleep(1)
                waited += 1

            if stop_event.is_set():
                print(f"🛑 Stop event detectado para câmera {camera_id}")
                break

            print(f"🔁 Preparando próxima gravação...\n")

        print(f"🛑 Agendador FINALIZADO para câmera {camera_id}")  # <-- aqui dentro!

    # Executa o job em uma thread separada
    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    return stop_event


@app.get("/debug/ts-files/{camera_id}")
def debug_ts_files(camera_id: int):
    """Endpoint para debug - lista arquivos .ts disponíveis"""
    try:
        ts_files = []
        for f in os.listdir(STREAMS_DIR):
            if f.startswith(f"camera_{camera_id}_") and f.endswith('.ts'):
                file_path = os.path.join(STREAMS_DIR, f)
                ts_files.append({
                    'name': f,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        
        return {
            "camera_id": camera_id,
            "directory": STREAMS_DIR,
            "ts_files": sorted(ts_files, key=lambda x: x['name']),
            "total_files": len(ts_files),
            "all_files_in_directory": os.listdir(STREAMS_DIR)
        }
    except Exception as e:
        return {"error": str(e), "directory": STREAMS_DIR}

def stop_auto_save(camera_id: int):
    """Para o salvamento automático de uma câmera"""
    if camera_id in save_events:
        print(f"🛑 Parando auto-save para câmera {camera_id}")
        save_events[camera_id].set()
        if camera_id in save_threads:
            save_threads[camera_id].join(timeout=5)
        
        save_events.pop(camera_id, None)
        save_threads.pop(camera_id, None)

def check_ip(ip, port=554, timeout=0.3):
    ip_str = str(ip)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip_str, port))
        sock.close()
        if result == 0:
            print(f"📡 Encontrada câmera em {ip_str}:{port}")
            return ip_str
    except:
        pass
    return None

def scan_cameras(ip_range="192.168.1.0/24", port=554):
    active_ips = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(check_ip, ip, port) for ip in ipaddress.IPv4Network(ip_range).hosts()]
        for future in concurrent.futures.as_completed(futures):
            ip = future.result()
            if ip:
                active_ips.append(ip)
    print("Câmeras encontradas:", active_ips)
    return active_ips

scan_cameras()

RTSP_TEMPLATE = "rtsp://admin:Arelsa.11@{ip}:554/h264/ch1s/main/av_stream"

def initialize_single_camera(ip: str):
    """Inicializa uma única câmera"""
    rtsp_url = RTSP_TEMPLATE.format(ip=ip)
    camera_id = ip

    try:
        print(f"📹 Iniciando câmera {camera_id}...")

        # Inicia streaming
        path = start_stream(camera_id, rtsp_url)
        time.sleep(5)

        # Inicia salvamento automático
        save_thread = schedule_auto_save(camera_id)

        active_streams[camera_id] = {
            'stream_url': f"/streams/camera_{camera_id}.m3u8",
            'save_thread': save_thread,
            'start_time': datetime.now()
        }

        # Inicia análise
        start_analysis_for_camera(camera_id, rtsp_url)
        active_analyses[camera_id] = True

        print(f"✅ Câmera {camera_id} inicializada com sucesso")
        return True

    except Exception as e:
        print(f"❌ Erro ao inicializar câmera {camera_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_hls_activity(interval=3):
    TIMEOUT = 30
    while True:
        for camera_id in list(active_streams):
            active = check_hls_activity(camera_id)
            if not active and time.time() - last_seen.get(camera_id, 0) > TIMEOUT:
                print(f"⚠️ Câmera {camera_id} sem atividade por {TIMEOUT}s. Encerrando...")
                del active_streams[camera_id]
        time.sleep(interval)

# Start monitor thread
threading.Thread(target=monitor_hls_activity, daemon=True).start()

def periodic_camera_scan(ip_range="192.168.1.0/24", interval_minutes=5):
    """
    Escaneia periodicamente a rede procurando por novas câmeras
    ou câmeras que ainda não estão ativas no active_streams
    """
    global known_cameras
    
    while not camera_scan_event.is_set():
        try:
            found_cameras = scan_cameras(ip_range)
            current_cameras = set(found_cameras)
            
            # Câmeras que ainda não estão ativas
            inactive_cameras = [ip for ip in current_cameras if ip not in active_streams]
            
            for ip in inactive_cameras:
                if initialize_single_camera(ip):
                    known_cameras.add(ip)
                    # initialize_single_camera já adiciona a câmera no active_streams
                    
        except Exception as e:
            print(f"❌ Erro no scanner periódico: {e}")
            import traceback
            traceback.print_exc()
        
        # Aguarda o intervalo (verificando stop event a cada segundo)
        print(f"⏳ Próximo scan em {interval_minutes} minutos...")
        for _ in range(interval_minutes * 60):
            if camera_scan_event.is_set():
                break
            time.sleep(1)
    
    print("🛑 Scanner periódico de câmeras finalizado")

def start_periodic_camera_scan(ip_range="192.168.1.0/24", interval_minutes=5):
    """Inicia o scanner periódico em thread separada"""
    camera_scan_event.clear()
    thread = threading.Thread(
        target=periodic_camera_scan,
        args=(ip_range, interval_minutes),
        daemon=True
    )
    thread.start()
    return thread


_initialized = False
_lock = threading.Lock()

def initialize_all_cameras_safe(ip_range="192.168.1.0/24"):
    global _initialized
    with _lock:
        if _initialized:
            print("⚠️ Inicialização das câmeras já executada, ignorando duplicata")
            return
        _initialized = True

    camera_ips = scan_cameras(ip_range)
    for ip in camera_ips:
        rtsp_url = RTSP_TEMPLATE.format(ip=ip)
        camera_id = ip
        try:
            print(f"📹 Iniciando câmera {camera_id}...")

            # Inicia streaming
            start_stream(camera_id, rtsp_url)

            # Salvar automaticamente
            save_thread = schedule_auto_save(camera_id)

            active_streams[camera_id] = {
                'stream_url': f"/streams/camera_{camera_id}.m3u8",
                'save_thread': save_thread,
                'start_time': datetime.now()
            }

            # Análise em segundo plano
            try:
                rtsp_url_dict = {"rtsp_address": rtsp_url}
                start_analysis_for_camera(camera_id, rtsp_url_dict)
                active_analyses[camera_id] = True
            except Exception as e:
                print(f"⚠️ Falha na análise da câmera {camera_id}: {e}")

            print(f"✅ Câmera {camera_id} inicializada com sucesso")
        except Exception as e:
            print(f"❌ Erro ao inicializar câmera {camera_id}: {e}")
    start_periodic_camera_scan()

@app.on_event("startup")
def startup_event():
    print("🚀 Inicializando todas as câmeras automaticamente...")
    threading.Timer(3.0, initialize_all_cameras_safe).start()
    start_cleanup_scheduler()
    threads = threading.enumerate()
    print(f"Threads ativas: {len(threads)}")
    for t in threads:
        print(f"- {t.name} (daemon={t.daemon})")

def stop_periodic_camera_scan():
    """Para o scanner periódico"""
    print("🛑 Parando scanner periódico de câmeras...")
    camera_scan_event.set()
    
@app.on_event("shutdown")
def shutdown_event():
    """Limpeza ao desligar o servidor"""
    print("🛑 Desligando sistema...")
    
    # Para o scanner de câmeras
    stop_periodic_camera_scan()
    
    # Para todos os agendadores de salvamento
    for camera_id in list(save_events.keys()):
        stop_auto_save(camera_id)
    
    # Para todas as câmeras
    for camera_id in list(active_streams.keys()):
        try:
            stop_stream(camera_id)
            stop_analysis_for_camera(camera_id)
        except Exception as e:
            print(f"Erro ao parar câmera {camera_id}: {e}")
    
    print("✅ Sistema desligado")

@app.get("/", response_class=HTMLResponse)
def index():
    return open("templates/index.html").read()

RECORDINGS_DIR = "recordings"

@app.get("/video")
def get_video(camera_id: str, timestamp: str):
    """
    Exemplo:
      /video?camera_id=192.168.1.4&timestamp=2025-10-16 09:31:34
    ou
      /video?camera_id=192.168.1.4&timestamp=2025-10-16_09-31-21
    """
    log_filename = f"recordings_log_camera_{camera_id}.txt"
    log_path = os.path.join(RECORDINGS_DIR, log_filename)

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log não encontrado para essa câmera")

    # Lê o log e busca a linha correspondente
    with open(log_path, "r") as f:
        lines = f.readlines()

    matched_file = None
    for line in lines:
        if timestamp in line:
            parts = line.strip().split("|")
            if len(parts) == 2:
                matched_file = parts[1].strip()
                break

    if not matched_file:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado para essa data/hora")

    video_path = os.path.join(RECORDINGS_DIR, matched_file)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Arquivo de vídeo não encontrado")

    return FileResponse(video_path, media_type="video/mp4")

@app.get("/cameras")
def list_cameras():
    cameras = []

    for camera_id, stream in active_streams.items():
        log_filename = f"recordings_log_camera_{camera_id}.txt"
        log_path = os.path.join(RECORDINGS_DIR, log_filename)

        # Lê o conteúdo do log se existir
        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_content = f.read()

        cameras.append({
            "id": camera_id,
            "status": "active",
            "objects_detected": len(class_counters.get(camera_id, {})),
            "auto_save": "active" if camera_id in save_events else "inactive",
            "log": log_content.strip().splitlines()  # lista de linhas do log
        })

    return cameras

@app.get("/camera/{camera_id}/stats")
def get_camera_stats(camera_id: str):
    if camera_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Câmera não está ativa")
    
    auto_save_status = "active" if camera_id in save_events else "inactive"
    
    return {
        "objects_detected": len(class_counters.get(camera_id, {})),
        "auto_save_status": auto_save_status,
        "object_breakdown": {
            class_name: count for class_name, count in class_counters.get(camera_id, {}).items()
        }
    }


@app.get("/start/{camera_ip}")
def start_camera(camera_ip: str):
    # Se a câmera já estiver ativa, retorna a URL
    if camera_ip in active_streams:
        return {
            "stream_url": active_streams[camera_ip]['stream_url']
        }
    
    # Monta a URL RTSP
    rtsp_url = f"rtsp://admin:Arelsa.11@{camera_ip}:554/h264/ch1s/main/av_stream"

    try:
        # Inicia o stream e registra no active_streams
        path = start_stream(camera_ip, rtsp_url)
        stream_url = f"/streams/camera_{camera_ip}.m3u8"

        active_streams[camera_ip] = {
            'stream_url': stream_url,
            'start_time': datetime.now()
        }

        return {
            "stream_url": stream_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao iniciar câmera: {str(e)}")


@app.get("/save/{camera_id}")
def save_last_seconds(camera_id: int, seconds: int = 10):
    if camera_id not in active_streams:
        raise HTTPException(status_code=404, detail="Câmera não está ativa")
    
    path = save_hls_as_mp4_fast(camera_id, last_seconds=seconds)
    if not path:
        raise HTTPException(status_code=404, detail="Nenhum segmento disponível")

    try:
        # save_video_record(
        #     camera_id=camera_id, 
        #     filename=os.path.basename(path),
        #     link=link_gcs,
        #     descricao=f"Gravação manual - últimos {seconds} segundos"
        # )
        return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar vídeo: {str(e)}")

@app.get("/status")
def system_status():
    auto_save_status = {}
    for camera_id in save_events:
        auto_save_status[camera_id] = "active"
    
    return {
        "active_cameras": list(active_streams.keys()),
        "total_objects_detected": sum(len(counters) for counters in class_counters.values()),
        "gpu_available": torch.cuda.is_available(),
        "auto_save_active": auto_save_status
    }

@app.get("/auto-save/status/{camera_id}")
def get_auto_save_status(camera_id: int):
    """Endpoint para verificar status do salvamento automático"""
    return {
        "camera_id": camera_id,
        "auto_save_active": camera_id in save_events,
        "camera_active": camera_id in active_streams,
        "last_check": datetime.now().isoformat()
    }

@app.get("/auto-save/interval/{camera_id}")
def set_auto_save_interval(camera_id: int, minutes: int = 1):
    """Endpoint para alterar o intervalo de salvamento automático (agora em minutos)"""
    if camera_id not in active_streams:
        raise HTTPException(status_code=404, detail="Câmera não está ativa")
    
    stop_auto_save(camera_id)
    
    schedule_auto_save(camera_id)
    
    return {"message": f"Intervalo de salvamento alterado para {minutes} minutos"}

@app.get("/debug/ts-files/{camera_id}")
def debug_ts_files(camera_id: int):
    """Endpoint para debug - lista arquivos .ts disponíveis"""
    camera_dir = os.path.join(STREAMS_DIR, f"camera_{camera_id}")
    if not os.path.exists(camera_dir):
        return {"error": f"Diretório não encontrado: {camera_dir}"}
    
    ts_files = []
    for f in os.listdir(camera_dir):
        if f.endswith('.ts'):
            file_path = os.path.join(camera_dir, f)
            ts_files.append({
                'name': f,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            })
    
    return {
        "camera_id": camera_id,
        "directory": camera_dir,
        "ts_files": sorted(ts_files, key=lambda x: x['name']),
        "total_files": len(ts_files)
    }

def parse_hikvision_xml(xml_content: str) -> dict:
    import xml.etree.ElementTree as ET
    from datetime import datetime

    root = ET.fromstring(xml_content)

    # Namespace Hikvision ANPR v2
    ns = {"hik": "http://www.hikvision.com/ver20/XMLSchema"}

    plate = root.findtext(".//hik:originalLicensePlate", default="", namespaces=ns)
    ip = root.findtext(".//hik:ipAddress", default="", namespaces=ns)
    vehicle_type = root.findtext(".//hik:vehicleType", default="", namespaces=ns)
    speed = root.findtext(".//hik:speed", default="", namespaces=ns)

    date_time = root.findtext(
        ".//hik:dateTime",
        default=datetime.now().isoformat(),
        namespaces=ns
    )

    return {
        "ip": ip.strip(),
        "plate_no": plate.strip(),
        "vehicle_type": vehicle_type.strip(),
        "speed": speed.strip(),
        "datetime": date_time.strip(),
    }


def is_valid_plate(plate: str) -> bool:
    return len(plate) >= 5 and any(c.isdigit() for c in plate)

stats = {}

@app.post("/camera/lpr/notify")
async def lpr_notify(
    request: Request,
    anpr_xml: UploadFile = File(None, alias="anpr.xml"),
    vehiclePicture: UploadFile = File(None, alias="vehiclePicture.jpg"),
    detectionPicture: UploadFile = File(None, alias="detectionPicture.jpg"),
):

    global stats
    print("teste")
    print("=" * 80)
    print("=" * 80)

    plate_detected = None
    should_process = False

    files_received = [name for name in request._form.keys()] if hasattr(request, "_form") else []
    print("📦 Arquivos recebidos:", files_received)

    if anpr_xml is not None:
        print("📄 XML recebido da câmera Hikvision")
        xml_content = (await anpr_xml.read()).decode("utf-8")
        print(xml_content)

        try:
            info = parse_hikvision_xml(xml_content)
            ip = info.get("ip")
            plate_no = info.get("plate_no") or "unknown"
            vehicle_type = info.get("vehicle_type") or "unknown"
            speed = info.get("speed") or "unknown"
            time = info.get("datetime") or datetime.now().isoformat()

            if is_valid_plate(plate_no):
                plate_detected = plate_no
                should_process = True
                stats["valid_plates"] += 1

                print("🎯" * 20)
                print(f"✅ PLACA DETECTADA: {plate_no}")
                print(f"📅 Data/Hora: {time}")
                print(f"🚙 Tipo: {vehicle_type}")
                print(f"📊 Velocidade: {speed}")
                print("🎯" * 20)

                # Registrar log
                with open("placas_detectadas.log", "a") as f:
                    f.write(f"{datetime.now()},{plate_no},{vehicle_type},{speed}\n")

            else:
                print(f"🚫 Ignorado — placa inválida ou desconhecida: '{plate_no}'")

        except Exception as e:
            print(f"❌ Erro ao processar XML: {e}")
            traceback.print_exc()
    else:
        print("teste2")

    if should_process and plate_detected:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if detectionPicture is not None:
            filename = os.path.join(FOTOS_DIR, f"veiculo_{plate_detected}_{timestamp}.jpg")
            with open(filename, "wb") as f:
                f.write(await vehiclePicture.read())
            print(f"📸 Foto do veículo salva: {filename}")
            video_path = save_hls_clip(ip, last_seconds=15)
            video_url = upload_video_to_gcs(video_path)

            save_ocorrencia(
                camera_id=cameraId,
                ip= ip,
                placa=plate_no,
                foto_url=filename,
                video_url=video_url,
                tipo="ocr"
            )


        if detectionPicture is not None:
            filename = os.path.join(FOTOS_DIR, f"detecao_{plate_detected}_{timestamp}.jpg")
            with open(filename, "wb") as f:
                f.write(await detectionPicture.read())
            print(f"📸 Foto da detecção salva: {filename}")

        print("🎉 EVENTO VÁLIDO PROCESSADO COM SUCESSO!")

    print("=" * 80)
    return JSONResponse({
        "status": "OK" if should_process else "IGNORED",
        "plate_detected": plate_detected,
        "processed": should_process,
        "stats": stats
    })

@app.get("/trigger-auto-save/{camera_id}")
def trigger_auto_save_now(camera_id: int):
    """Força uma execução imediata do auto-save (para testes)"""
    if camera_id not in active_streams:
        raise HTTPException(status_code=404, detail="Câmera não está ativa")
    
    # Simula o que o agendador faria
    output_path = save_hls_as_mp4_fast(camera_id)
    if output_path and os.path.exists(output_path):
        link_gcs = upload_video_to_gcs(output_path)
        save_video_record(
            camera_id=camera_id,
            filename=os.path.basename(output_path),
            link=link_gcs,
            descricao=f"Gravação manual trigger - {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )
        return {"message": "Auto-save triggered", "file": output_path, "gcs_url": link_gcs}
    else:
        return {"message": "Nenhum vídeo gerado", "file": output_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")