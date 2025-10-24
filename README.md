# Tópicos-Especiais-Computação-II
Atividade Prática: Ciclo 1
## Tópicos Especiais em Computação II — Atividade Prática (Ciclo 1)

Este projeto implementa um classificador de imagens (Pedra / Papel / Tesoura) treinado no Google Teachable Machine e executado localmente em Python usando TensorFlow e OpenCV.

O objetivo é demonstrar o fluxo completo: coleta de imagens, treinamento no Teachable Machine, exportação do modelo (Keras) e execução do classificador em um computador (webcam, vídeo ou pasta de imagens).

### O que tem neste repositório

- `classificador.py` — script principal que carrega `keras_model.h5` e `labels.txt` e mostra predições em tempo real (webcam) ou processando um arquivo de vídeo / pasta de imagens.
- `capture_images.py` — utilitário para capturar imagens da webcam e construir um dataset local (pasta `data/<label>`).
- `keras_model.h5` — modelo Keras exportado do Teachable Machine (pode estar grande; se não estiver no repositório, coloque-o na pasta do projeto).
- `labels.txt` — rótulos exportados pelo Teachable Machine.
- `README_RUN.md` — instruções detalhadas de execução e dicas para gravação do vídeo de demonstração.
- `run_test.ps1` — script PowerShell que cria/usa um venv, instala dependências e executa o classificador com `meu_teste.mp4`.
- `pack_submission.ps1` — script PowerShell que empacota `classificador.py`, `keras_model.h5` e `labels.txt` em `submission.zip` para envio.
- `video_script.txt` — roteiro curto sugerido para gravar sua demonstração.

### Requisitos

- Python 3.8+ (o projeto foi testado em Python 3.10/3.11/3.12). Recomenda-se usar um virtual environment (venv).
- Pacotes Python: `tensorflow`, `opencv-python`.

### Como executar (PowerShell)

1. Abra o PowerShell e vá para a pasta do projeto:

```powershell
cd "C:\Users\joaov\OneDrive\Documentos\Developer\T-picos-Especiais-Computa--o-II"
```

2. Criar e ativar venv (opcional, recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependências (se não usar `run_test.ps1`):

```powershell
python -m pip install --upgrade pip
pip install tensorflow opencv-python
```

4. Executar o classificador com um arquivo de vídeo (se não tiver webcam):

```powershell
python classificador.py .\meu_teste.mp4
```

Você também pode executar sem argumento (tenta abrir webcam `0`):

```powershell
python classificador.py
```

Ou apontar para uma pasta de imagens:

```powershell
python classificador.py "C:\caminho\para\pasta_de_imagens"
```

Controles na janela do OpenCV enquanto o script roda:
- `q` — sair
- `c` — salvar screenshot do frame atual
- `n` — avançar para a próxima imagem (quando a fonte for uma pasta de imagens)

### Como gerar o ZIP de submissão

Use o script PowerShell `pack_submission.ps1` para criar `submission.zip` com os três arquivos exigidos:

```powershell
.\pack_submission.ps1
```

### Dicas rápidas para o vídeo de demonstração (nota máxima)

- Mostre coleta de dados (Teachable Machine ou pasta `data/`). Capture 100–200 imagens por classe e varie ângulo, iluminação e distância.
- Mostre o treinamento e a mensagem "Training complete" no Teachable Machine.
- Exporte o modelo como Keras (converted_keras.zip) e coloque `keras_model.h5` e `labels.txt` na pasta do projeto.
- Execute `python classificador.py meu_teste.mp4` (ou com webcam) e grave a tela mostrando as predições e as porcentagens.
- Mostre casos corretos e um caso em que o modelo erra; explique como melhorar (mais imagens, variação).

### Problemas comuns

- Se o script reclamar que `keras_model.h5` ou `labels.txt` não foram encontrados, verifique se estão no mesmo diretório que `classificador.py`.
- Se houver erro ao importar `tensorflow` ou `cv2`, instale os pacotes dentro do venv com `pip install tensorflow opencv-python`.
- Se o vídeo não abrir (codec), converta para MP4 H.264 usando um conversor (ex: ffmpeg).

Se precisar, eu adapto os scripts ou o README para seu caso específico (ex.: instruções para Conda, versões específicas do TensorFlow, etc.).

Boa sorte na entrega — se quiser, eu já crio o `submission.zip` por você quando confirmar que os três arquivos estão na pasta.
