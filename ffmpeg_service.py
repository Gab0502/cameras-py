import subprocess
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import glob
import time
import re

STREAMS_DIR = "streams"
RECORDINGS_DIR = "recordings"
os.makedirs(STREAMS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

processes = {}
executor = ThreadPoolExecutor(max_workers=4)  # Para conversões paralelas
TIMEOUT = 30  # segundos

processes = {}  # camera_id: Popen
last_seen = {}  # camera_id: timestamp do último frame

def check_hls_activity(camera_id: int):
    """Atualiza heartbeat se a câmera estiver enviando novos segmentos"""
    m3u8_path = os.path.join(STREAMS_DIR, f"camera_{camera_id}.m3u8")
    if not os.path.exists(m3u8_path):
        return False
    
    last_mod_time = os.path.getmtime(m3u8_path)
    
    if camera_id not in last_seen:
        last_seen[camera_id] = last_mod_time
        return True
    
    if last_mod_time != last_seen[camera_id]:
        last_seen[camera_id] = last_mod_time
        update_heartbeat(camera_id)
        return True
    
    return False

def save_hls_clip(camera_id: int, seconds=30, filename=None):
    """
    Gera clip MP4 com os ÚLTIMOS X segundos do buffer HLS
    SOLUÇÃO DEFINITIVA: Ordena por data de modificação real do arquivo
    """
    
    # Pega arquivos TS existentes
    pattern = os.path.join(STREAMS_DIR, f"camera_{camera_id}_*.ts")
    ts_files = glob.glob(pattern)
    
    if not ts_files:
        print(f"⚠️ Nenhum segmento encontrado para câmera {camera_id}")
        return None
    
    # 🔥 ORDENA POR DATA DE MODIFICAÇÃO DO SISTEMA DE ARQUIVOS
    # (não pelo nome, que é reciclado)
    ts_files_sorted = sorted(ts_files, key=lambda f: os.path.getmtime(f))
    
    # Debug: mostra os timestamps reais
    if len(ts_files_sorted) > 0:
        primeiro_mod = datetime.fromtimestamp(os.path.getmtime(ts_files_sorted[0]))
        ultimo_mod = datetime.fromtimestamp(os.path.getmtime(ts_files_sorted[-1]))
        print(f"📂 Total: {len(ts_files_sorted)} segmentos")
        print(f"   Mais antigo: {os.path.basename(ts_files_sorted[0])} - {primeiro_mod.strftime('%H:%M:%S')}")
        print(f"   Mais recente: {os.path.basename(ts_files_sorted[-1])} - {ultimo_mod.strftime('%H:%M:%S')}")
    
    # Calcula quantos segmentos precisa (cada segmento = 2s)
    hls_time = 2
    segments_needed = max(1, int(seconds / hls_time))
    
    # Pega os ÚLTIMOS N segmentos (por tempo de modificação)
    recent_segments = ts_files_sorted[-segments_needed:]
    
    print(f"🎯 Usando {len(recent_segments)} segmentos mais recentes")
    for seg in recent_segments[:3]:  # Mostra os 3 primeiros
        mod_time = datetime.fromtimestamp(os.path.getmtime(seg))
        print(f"   {os.path.basename(seg)} - modificado: {mod_time.strftime('%H:%M:%S.%f')[:-3]}")
    if len(recent_segments) > 3:
        print(f"   ... e mais {len(recent_segments)-3} segmentos")
    
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"camera_{camera_id}_clip_{timestamp}.mp4"
    
    output_path = os.path.join(RECORDINGS_DIR, filename)
    
    # Cria lista temporária para concat
    list_file = os.path.join(STREAMS_DIR, f"temp_clip_{camera_id}_{int(time.time()*1000)}.txt")
    with open(list_file, "w") as f:
        for ts in recent_segments:
            f.write(f"file '{os.path.abspath(ts)}'\n")
    
    # FFmpeg com concat
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]
    
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=10
        )
        
        os.remove(list_file)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        duration_real = len(recent_segments) * hls_time
        print(f"✅ Clip {duration_real}s salvo: {output_path} ({size_mb:.2f} MB)")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout ao gerar clip da câmera {camera_id}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return None
        
    except subprocess.CalledProcessError as e:
        print("❌ Erro FFmpeg clip:")
        print(e.stderr.decode())
        if os.path.exists(list_file):
            os.remove(list_file)
        return None

def start_stream(camera_id: int, cam: str):
    """Inicia stream HLS com configurações otimizadas"""
    output_path = os.path.join(STREAMS_DIR, f"camera_{camera_id}.m3u8")
    
    # Se já existe stream ativo, para antes de reiniciar
    if camera_id in processes:
        stop_stream(camera_id)
    
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", cam,
        
        # Otimizações de input
        "-fflags", "+genpts+discardcorrupt",
        "-use_wallclock_as_timestamps", "1",
        "-avoid_negative_ts", "make_zero",
        
        # Video encoding otimizado
        "-c:v", "copy",  # sem re-encode, mais rápido
        "-bsf:v", "h264_mp4toannexb",

        
        # HLS settings otimizados
        "-f", "hls",
        "-hls_time", "2",                       # duração de cada segmento
        "-hls_list_size", "1800",               # 1 hora de segmentos (1800 × 2s)
        "-hls_flags", "delete_segments+append_list+omit_endlist",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", os.path.join(STREAMS_DIR, f"camera_{camera_id}_%05d.ts"),
        
        # Performance
        "-threads", "2",
        "-max_muxing_queue_size", "1024",
        
        "-y",
        output_path
    ]
    
    # Inicia processo único
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL
    )
    
    # Guarda o processo e marca último "ping" da câmera
    processes[camera_id] = proc
    last_seen[camera_id] = time.time()
    
    return output_path

def update_heartbeat(camera_id: int):
    if camera_id in last_seen:
        last_seen[camera_id] = time.time()
        
def stop_stream(camera_id: int):
    """Para o stream de forma segura"""
    if camera_id in processes:
        proc = processes[camera_id]
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        del processes[camera_id]


def save_hls_as_mp4_fast(camera_id: int, filename=None, last_seconds=None, delete_ts=False):
    """
    Versão OTIMIZADA - usa codec copy (sem re-encode)
    Até 10x mais rápido que a versão anterior
    delete_ts: se True, apaga os TS já usados após gerar o MP4
    """
    pattern = os.path.join(STREAMS_DIR, f"camera_{camera_id}_*.ts")
    ts_files = sorted(glob.glob(pattern))
    
    if not ts_files:
        print(f"⚠️ Nenhum arquivo TS encontrado para câmera {camera_id}")
        return None
    
    if last_seconds:
        hls_time = 2
        count = max(1, int(last_seconds / hls_time))
        ts_files = ts_files[-count:]
    
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"camera_{camera_id}_{timestamp}.mp4"
    
    output_path = os.path.join(RECORDINGS_DIR, filename)
    
    list_file = os.path.join(STREAMS_DIR, f"temp_list_{camera_id}.txt")
    with open(list_file, "w") as f:
        for ts in ts_files:
            f.write(f"file '{os.path.abspath(ts)}'\n")
    
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", 
        output_path
    ]
    
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )
        os.remove(list_file)

        if delete_ts:
            for ts in ts_files:
                try:
                    print(ts)
                    os.remove(ts)
                except Exception as e:
                    print(f"⚠️ Não foi possível remover {ts}: {e}")

        # Log
        log_filename = f"recordings_log_camera_{camera_id}.txt"
        log_path = os.path.join(RECORDINGS_DIR, log_filename)
        with open(log_path, "a") as log:
            log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {filename}\n")
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ MP4 gerado: {output_path} ({file_size:.2f} MB)")
        return output_path
        
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout ao processar vídeo da câmera {camera_id}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro FFmpeg: {e.stderr.decode()}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return None



async def save_hls_as_mp4_async(camera_id: int, filename=None, last_seconds=None):
    """Versão assíncrona para uso em FastAPI"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        save_hls_as_mp4_fast,
        camera_id,
        filename,
        last_seconds
    )


def save_multiple_cameras_parallel(camera_ids: list, last_seconds=None):
    """Salva vídeos de múltiplas câmeras em paralelo"""
    from concurrent.futures import as_completed
    
    results = {}
    futures = {
        executor.submit(save_hls_as_mp4_fast, cam_id, None, last_seconds): cam_id
        for cam_id in camera_ids
    }
    
    for future in as_completed(futures):
        cam_id = futures[future]
        try:
            result = future.result(timeout=60)
            results[cam_id] = result
        except Exception as e:
            print(f"❌ Erro ao processar câmera {cam_id}: {e}")
            results[cam_id] = None
    
    return results


# ===== OPÇÃO AVANÇADA: Usando FFmpeg em modo streaming =====
def save_hls_streaming(camera_id: int, duration_seconds: int):
    """
    Alternativa: captura direto do RTSP sem passar por HLS
    Útil para gravações longas e programadas
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"camera_{camera_id}_{timestamp}_direct.mp4"
    output_path = os.path.join(RECORDINGS_DIR, filename)
    
    # Precisaria da URL RTSP original - você pode armazenar em um dict
    # rtsp_url = get_camera_rtsp_url(camera_id)
    
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", "rtsp_url_aqui",
        "-t", str(duration_seconds),
        "-c", "copy",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path


# ===== OPÇÃO COM SEGMENTAÇÃO INTELIGENTE =====
def smart_segment_manager(camera_id: int, max_segments=1800):
    """
    Mantém apenas os últimos N segmentos em disco
    Evita crescimento infinito de arquivos
    """
    pattern = os.path.join(STREAMS_DIR, f"camera_{camera_id}_*.ts")
    ts_files = sorted(glob.glob(pattern))
    
    if len(ts_files) > max_segments:
        files_to_delete = ts_files[:-max_segments]
        for f in files_to_delete:
            try:
                os.remove(f)
            except OSError:
                pass
        print(f"🗑️ Removidos {len(files_to_delete)} segmentos antigos da câmera {camera_id}")