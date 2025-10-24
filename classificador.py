# -*- coding: utf-8 -*-
"""
Classificador Pedra/Papel/Tesoura (Teachable Machine)
- Fontes: webcam (índice), arquivo de vídeo (.mp4/.avi...), ou pasta de imagens.
- Visualização: letterbox (mostra quadro inteiro), rotação com tecla 'r'.
- Overlay: TOP-K predições com suavização (média móvel).
- Teclas:
    q = sair
    c = salvar screenshot
    n = próxima imagem (apenas no modo "pasta de imagens")
    r = rotacionar (0 -> 90 -> 180 -> 270 -> 0)
"""

import os
import sys
import cv2
import numpy as np
from collections import deque

# Evita alguns bugs/avisos de otimizações OneDNN em Windows/CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import DepthwiseConv2D as _KDepthwise
except Exception:
    print("Erro ao importar TensorFlow/Keras. Verifique se 'tensorflow' está instalado.")
    print("Ex.: pip install tensorflow")
    raise

# -------- Configurações --------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

INPUT_SIZE = (224, 224)     # TM (Keras export) normalmente usa 224x224
SMOOTHING_WINDOW = 5        # janela da média móvel (estabiliza predições)
TOP_K = 3                   # quantas classes exibir (TOP-K)

# Limites da janela de exibição (ajuste se quiser)
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

# -------- Utilitários de exibição --------
def letterbox(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H, color=(0, 0, 0)):
    """Redimensiona mantendo proporção e insere em um canvas (sem cortar)."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.full((max_h, max_w, 3), color, dtype=np.uint8)
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((max_h, max_w, 3), color, dtype=np.uint8)
    y0 = (max_h - new_h) // 2
    x0 = (max_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

# -------- Funções de modelo/labels --------
def load_labels(path: str):
    """Carrega rótulos do Teachable Machine e remove prefixos tipo '0 ' / '0. '."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f.readlines() if ln.strip()]

    labels = []
    for ln in raw:
        # Remove primeiro token se parecer índice/prefixo
        if " " in ln:
            left, right = ln.split(" ", 1)
            if left.replace(".", "").isdigit() or left.lower() in {"class", "classe"}:
                ln = right.strip()
        labels.append(ln)
    return labels

def preprocess_frame(frame, size=INPUT_SIZE):
    """Redimensiona e normaliza frame para [-1, 1] como no export do TM."""
    img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    img = (img / 127.5) - 1.0
    return np.expand_dims(img, axis=0)

def draw_topk(frame, labels, probs, top_k=TOP_K, origin=(10, 35)):
    """Desenha as TOP-K classes com suas probabilidades no frame."""
    top_idx = np.argsort(probs)[::-1][:top_k]
    y0 = origin[1]
    for i, idx in enumerate(top_idx):
        name = labels[idx] if idx < len(labels) else f"Classe {idx}"
        conf = float(probs[idx]) * 100.0
        text = f"{i+1}. {name}: {conf:.1f}%"
        cv2.putText(frame, text, (origin[0], y0 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 230, 40), 2, cv2.LINE_AA)

# -------- Abertura de fonte (vídeo / imagens / webcam) --------
def open_video_source(source: str):
    """
    Ordem de tentativa:
      1) arquivo de vídeo (caminho absoluto/relativo)
      2) pasta de imagens
      3) índice de câmera (0,1,...)
    Retorna (mode, handle):
      - 'video'  + cv2.VideoCapture
      - 'images' + [lista de caminhos]
      - 'camera' + cv2.VideoCapture
      - (None, None) se nada abrir
    """
    abspath = os.path.abspath(source)

    # 1) Arquivo de vídeo
    if os.path.isfile(source) or os.path.isfile(abspath):
        path = source if os.path.isfile(source) else abspath
        print(f"[INFO] Tentando abrir arquivo de vídeo: {path}")
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path, cv2.CAP_ANY)
        if cap.isOpened():
            print("[OK] Vídeo aberto com sucesso.")
            return 'video', cap
        else:
            print("[ERRO] Não foi possível abrir o arquivo de vídeo. "
                  "Tente mover/renomear para um caminho sem acentos/espaços.")

    # 2) Pasta de imagens
    if os.path.isdir(source) or os.path.isdir(abspath):
        folder = source if os.path.isdir(source) else abspath
        print(f"[INFO] Lendo pasta de imagens: {folder}")
        imgs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if imgs:
            print(f"[OK] {len(imgs)} imagens encontradas.")
            return 'images', imgs
        else:
            print("[ERRO] Pasta encontrada, mas sem imagens .png/.jpg/.jpeg/.bmp.")

    # 3) Índice de câmera
    try:
        idx = int(source)
        print(f"[INFO] Tentando abrir webcam índice {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print("[OK] Webcam aberta.")
            return 'camera', cap
        else:
            cap.release()
            print("[ERRO] Webcam não abriu (índice inválido).")
    except ValueError:
        pass

    return None, None

# -------- Principal --------
def main(source="0"):
    # Verifica arquivos essenciais
    if not os.path.exists(MODEL_PATH):
        print(f"Modelo não encontrado: {MODEL_PATH}. Coloque 'keras_model.h5' na mesma pasta do script.")
        sys.exit(1)
    if not os.path.exists(LABELS_PATH):
        print(f"Labels não encontrado: {LABELS_PATH}. Coloque 'labels.txt' na mesma pasta do script.")
        sys.exit(1)

    print("Carregando modelo... (pode demorar um pouco)")

    # Wrapper compatível p/ alguns exports do TM que passam 'groups' em DepthwiseConv2D
    class _CompatibleDepthwise(_KDepthwise):
        def __init__(self, *args, groups=None, **kwargs):
            super().__init__(*args, **kwargs)

    try:
        model = load_model(MODEL_PATH, compile=False,
                           custom_objects={'DepthwiseConv2D': _CompatibleDepthwise})
    except Exception:
        # fallback: tenta sem custom_objects
        model = load_model(MODEL_PATH, compile=False)

    labels = load_labels(LABELS_PATH)
    print(f"Modelo carregado. Classes: {labels}")

    # Abre a fonte solicitada
    mode, handle = open_video_source(str(source))
    if mode is None:
        print("Não foi possível abrir a fonte de vídeo.")
        print("Uso:")
        print("  Webcam padrão:     python classificador.py")
        print("  Webcam índice 1:   python classificador.py 1")
        print("  Vídeo arquivo:     python classificador.py caminho\\para\\meu.teste.mp4")
        print("  Pasta de imagens:  python classificador.py caminho\\para\\imagens")
        sys.exit(1)

    cap = handle if mode in ('camera', 'video') else None
    images_list = handle if mode == 'images' else None
    img_idx = 0

    # buffer p/ suavização
    probs_buffer = deque(maxlen=SMOOTHING_WINDOW)

    # Sincroniza com FPS do vídeo, quando disponível
    wait_ms = 1
    if cap is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            wait_ms = max(1, int(1000 / fps))

    # Janela redimensionável + estado de rotação
    window_name = "Classificador | Teachable Machine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_MAX_W, DISPLAY_MAX_H)
    rotate_state = 0  # 0, 90, 180, 270

    while True:
        # lê frame
        if cap is not None:  # vídeo/câmera
            ok, frame = cap.read()
            if not ok:
                if mode == 'video':
                    print("Fim do arquivo de vídeo.")
                else:
                    print("Não foi possível ler frame da fonte.")
                break
        else:  # pasta de imagens
            if img_idx >= len(images_list):
                print("Fim das imagens na pasta.")
                break
            img_path = images_list[img_idx]
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Não foi possível ler a imagem: {img_path}")
                img_idx += 1
                continue

        # Rotação opcional (útil para vídeos verticais do celular)
        if rotate_state == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_state == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate_state == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # predição
        inp = preprocess_frame(frame, INPUT_SIZE)
        prediction = model.predict(inp, verbose=0)[0]  # vetor (logits ou probs)
        probs = prediction / (np.sum(prediction) + 1e-8)  # normaliza p/ distribuição

        # suavização
        probs_buffer.append(probs)
        avg_probs = np.mean(np.stack(probs_buffer, axis=0), axis=0)

        # desenha overlay no frame original
        draw_topk(frame, labels, avg_probs, top_k=TOP_K, origin=(10, 35))
        cv2.putText(frame,
                    "q: sair | c: screenshot | n: prox imagem (modo pasta) | r: rotacionar",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # cria versão letterbox para visualizar sem corte na janela
        frame_disp = letterbox(frame, DISPLAY_MAX_W, DISPLAY_MAX_H)
        cv2.imshow(window_name, frame_disp)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            out_name = f"screenshot_{cv2.getTickCount()}.png"
            cv2.imwrite(out_name, frame)
            print(f"Screenshot salvo: {out_name}")
        elif key == ord('n') and images_list is not None:
            img_idx += 1
        elif key == ord('r'):
            rotate_state = (rotate_state + 90) % 360

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Aceita: índice da câmera (ex.: "0"), caminho de vídeo (ex.: "meu.teste.mp4")
    # ou pasta de imagens (ex.: ".\\imagens")
    source = "0"  # padrão = webcam 0
    if len(sys.argv) > 1:
        source = sys.argv[1]  # mantém string; open_video_source decide
    main(source)
