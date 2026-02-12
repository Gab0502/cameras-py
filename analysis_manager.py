import threading
import queue
import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sort.sort import *
import torch
import os
from dotenv import load_dotenv
load_dotenv()
import warnings

active_streams = {}
active_analyses = {}

FRAME_QUEUE_SIZE = 8
VEHICLES = [2, 3, 5, 7]
PEOPLE = [0]
COCO_CLASS_NAMES = {
    0: 'person',
    2: 'car', 
    3: 'motorbike',
    5: 'bus',
    7: 'truck'
}

def setup_torch_safe_globals():
    """Configura as classes seguras para carregar modelos YOLO"""
    safe_classes = [
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.head.Detect',
        'ultralytics.models.yolo.detect.DetectionPredictor',
        'ultralytics.models.yolo.detect.DetectionValidator',
        'ultralytics.models.yolo.detect.DetectionTrainer',
        'torch.nn.modules.container.Sequential',
        'torch.nn.modules.container.ModuleList',
        'torch.nn.modules.conv.Conv2d',
        'torch.nn.modules.batchnorm.BatchNorm2d',
        'torch.nn.modules.activation.SiLU',
        'torch.nn.modules.pooling.AdaptiveAvgPool2d',
        'torch.nn.modules.linear.Linear',
        'torch.nn.modules.dropout.Dropout',
        'torch.nn.parameter.Parameter',
        'collections.OrderedDict'
    ]
    
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass
    
    import torch.nn as nn
    torch_classes = [
        nn.Sequential, nn.ModuleList, nn.Conv2d, nn.BatchNorm2d, 
        nn.SiLU, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout
    ]
    
    try:
        torch.serialization.add_safe_globals(torch_classes)
    except Exception as e:
        print(f"⚠️  Aviso ao adicionar classes seguras: {e}")

def load_yolo_model_safe(model_path):
    """Carrega modelo YOLO de forma segura, lidando com PyTorch 2.6+"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo {model_path} não encontrado")
    
    try:
        setup_torch_safe_globals()
        model = YOLO(model_path)
        return model
    except Exception as e1:
        print(f"⚠️  Tentativa 1 falhou: {e1}")
        
        try:
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*weights_only.*")
                model = YOLO(model_path)
            
            torch.load = original_load
            
            return model
            
        except Exception as e2:
            print(f"❌ Erro final ao carregar {model_path}: {e2}")
            raise e2


try:
    coco_model = load_yolo_model_safe('./yolov8n.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        coco_model.to(device)
    
except Exception as e:
    print(f"❌ Erro crítico ao carregar modelo: {e}")
    print("💡 Possíveis soluções:")
    print("   1. Verifique se o arquivo yolov8n.pt existe no diretório")
    print("   2. Reinstale ultralytics: pip install --upgrade ultralytics")
    print("   3. Use uma versão anterior do PyTorch se necessário")
    exit(1)

class_counters = defaultdict(lambda: defaultdict(int))
tracked_objects = defaultdict(set)
mot_tracker_dict = {}


class VideoCaptureThread(threading.Thread):
    def __init__(self, camera_id, rtsp_url):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url["rtsp_address"]
        self.cap = cv2.VideoCapture(rtsp_url["rtsp_address"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = True
        self.frame_count = 0

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.camera_id}] Falha na captura. Tentando reconectar...")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue

            try:
                frame_height, frame_width = frame.shape[:2]
                h = int(frame_height * 0.26)
                w = frame_width // 2
                x = int((frame_width - w))
                y = int(frame_height * 0.2)
                frame_cropped = frame[y:y+h, x:x+w]

                # 💾 Salva apenas o primeiro frame para ver se está funcionando
                if self.frame_count == 0:
                    os.makedirs("frames_teste", exist_ok=True)
                    cv2.imwrite(f"frames_teste/camera_{self.camera_id}_teste.jpg", frame_cropped)
                    print(f"[{self.camera_id}] Frame salvo em frames_teste/camera_{self.camera_id}_teste.jpg")

                self.queue.put((frame_cropped, self.frame_count), timeout=0.1)
                self.frame_count += 1

            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.put((frame_cropped, self.frame_count), timeout=0.1)
                    self.frame_count += 1
                except (queue.Empty, queue.Full):
                    pass

    def read(self):
        try:
            return self.queue.get(timeout=0.1)
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


capture_threads = {}
detection_threads = {}


class DetectionThread(threading.Thread):
    def __init__(self, camera_id, capture_thread):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.capture_thread = capture_thread
        if camera_id not in mot_tracker_dict:
            mot_tracker_dict[camera_id] = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.mot_tracker = mot_tracker_dict[camera_id]        
        self.running = True
        self.last_stat_print = time.time()
        self.frame_skip = 1

    def run(self):
        frame_counter = 0        
        while self.running:
            frame, frame_count = self.capture_thread.read()
            if frame is None:
                if not self.capture_thread.running:
                    break
                time.sleep(0.01)
                continue
            
            frame_counter += 1
            if frame_counter % self.frame_skip != 0:
                continue

            with torch.no_grad():
                try:                    
                    detections = coco_model(frame, verbose=False, conf=0.3)[0]                    
                    detections_ = []
                    for det in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = det
                        class_id = int(class_id)
                        if class_id in VEHICLES + PEOPLE and score > 0.3:
                            detections_.append([x1, y1, x2, y2, score])                    
                    track_ids = []
                    if detections_:
                        track_ids = self.mot_tracker.update(np.asarray(detections_))
                        self.update_counters(detections, track_ids)
                        
                except Exception as e:
                    continue          
            
            if time.time() - self.last_stat_print > 30:
                self.print_stats()
                self.last_stat_print = time.time()

    def update_counters(self, detections, track_ids):
        """Atualiza contadores de objetos detectados"""
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)            
            if class_id not in VEHICLES + PEOPLE:
                continue               
            
            track_id = None 
            
            for t in track_ids:
                if (abs(t[0]-x1) < 50 and abs(t[1]-y1) < 50 and 
                    abs(t[2]-x2) < 50 and abs(t[3]-y2) < 50):
                    track_id = int(t[4])
                    break            
            
            if track_id is not None:
                object_id = f"{track_id}_{class_id}"
                if object_id not in tracked_objects[self.camera_id]:
                    class_name = COCO_CLASS_NAMES.get(class_id, f'class_{class_id}')
                    class_counters[self.camera_id][class_name] += 1
                    tracked_objects[self.camera_id].add(object_id)

    def detect_plates(self, frame, track_ids, frame_count):
        """Função mantida para compatibilidade - não faz nada"""
        pass

    def print_stats(self):
        """Imprime estatísticas de contagem"""
        counters = class_counters[self.camera_id]
        if counters:
            stats = ", ".join([f"{name}: {count}" for name, count in sorted(counters.items())])
            print(f"[{self.camera_id}] Contagem: {stats}")

    def stop(self):
        self.running = False


def start_analysis_for_camera(camera_id, rtsp_url):
    """Inicia análise para uma câmera específica"""
    print("começando analise")
    if camera_id in capture_threads:
        return False
    try:
        print(f"Iniciando análise para câmera {camera_id}")
        cap_thread = VideoCaptureThread(camera_id, rtsp_url)
        det_thread = DetectionThread(camera_id, cap_thread)
        cap_thread.start()
        time.sleep(2)  
        det_thread.start()
        capture_threads[camera_id] = cap_thread
        detection_threads[camera_id] = det_thread
        return True
    except Exception as e:
        print(f"Erro ao iniciar análise: {e}")
        return False


def stop_analysis_for_camera(camera_id):
    """Para análise de uma câmera específica"""
    stopped = False
    if camera_id in detection_threads:
        detection_threads[camera_id].stop()
        detection_threads.pop(camera_id, None)
        stopped = True
    if camera_id in capture_threads:
        capture_threads[camera_id].stop()
        capture_threads.pop(camera_id, None)
        stopped = True
    return stopped