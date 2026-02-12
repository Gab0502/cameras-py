import json
import os
from sqlalchemy import create_engine, text
from datetime import datetime
DATABASE_URL = os.getenv("DB_URL")  # coloque seu DB_URL no env

engine = create_engine("", connect_args={"sslmode": "require"})

def get_cameras():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id_cam, lat FROM camera_points"))
        return [{"id": row.id_cam, "nome": row.lat} for row in result]

def get_camera_rtsp(camera_id: int):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM camera_points WHERE id_cam = :id"), {"id": camera_id})
        row = result.fetchone()
        if row:
            row_dict = dict(row._mapping)  # Converte Row para dict
            # Se quiser como JSON string:
            return row_dict  # default=str para datetime/Decimal
        return None
    
    
def save_video_record(camera_id: int, filename: str, link: str, descricao: str = ""):
    tamanho = os.path.getsize(filename)  # tamanho em bytes
    with engine.connect() as conn:
        query = text("""
            INSERT INTO historico_filmagem (camera_id, tamanho_clipe, "name", descricao, link, createdat)
            VALUES (:camera_id, :tamanho, :name, :descricao, :link, :createdat)
        """)
        conn.execute(query, {
            "camera_id": camera_id,
            "tamanho": str(tamanho),
            "name": os.path.basename(filename),
            "descricao": descricao,
            "link": link,
            "createdat": datetime.now() 
        })
        conn.commit()

def save_ocorrencia(
    camera_id: int,
    ip: str,
    placa: str,
    video_url: str,
    tipo: str,
    foto_url: str = None,
    sentido: str = None,
    velocidade: float = None
):
    try:
        with engine.begin() as conn:
            query = text("""
                INSERT INTO ocorrencias (
                    id_cam, placa, data_ocorrencia,
                    sentido_veiculo, velocidade,
                    foto_url, video_url, tipo, internal_ip
                )
                VALUES (
                    :id_cam, :placa, :data_ocorrencia,
                    :sentido_veiculo, :velocidade,
                    :foto_url, :video_url, :tipo, :internal_ip
                )
            """)

            result = conn.execute(query, {
                "id_cam": camera_id,
                "placa": placa,
                "data_ocorrencia": datetime.now(),
                "sentido_veiculo": sentido,
                "velocidade": velocidade,
                "foto_url": foto_url,
                "video_url": video_url,
                "tipo": tipo,
                "internal_ip": ip
            })

            print("✅ INSERT executado — linhas afetadas:", result.rowcount)

    except SQLAlchemyError as e:
        print("❌ ERRO SQLAlchemy:")
        print(e)
        traceback.print_exc()