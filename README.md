# YOLOv5 Object Detection - Projeto GRUPO1-HACKATON

Este README descreve todo o processo de configuração do ambiente, instalação de dependências e execução do projeto de detecção de objetos usando YOLOv5.

## Requisitos

- Python 3.13
- Git
- Virtualenv (recomendado)

## Passo 1: Clonar o Repositório do YOLOv5

```bash
# Clone o repositório do YOLOv5
git clone https://github.com/ultralytics/yolov5.git
```

Após o clone, mova a pasta `runs` para dentro da pasta `yolov5`:

```bash
mv runs yolov5/
```

## Passo 2: Criar o Ambiente Virtual

Crie e ative o ambiente virtual:

```bash
python3 -m venv env
source env/bin/activate
```

## Passo 3: Instalar as Dependências

Instale as bibliotecas necessárias:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib pandas seaborn
pip install python-dotenv
```

## Passo 4: Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
touch .env
```

Adicione o seguinte conteúdo ao arquivo `.env`:

```env
EMAIL_SENDER=grupo1hacka@gmail.com
EMAIL_PASSWORD=gszu cwtn tkwv dbwn
EMAIL_RECEIVER=grupo1hacka@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

## Passo 5: Ajustar o Código

### Correção de Imports no `objectDetection.py`

Adicione o seguinte código no início do `objectDetection.py` para corrigir problemas de compatibilidade entre Windows e macOS:

```python
import pathlib
import sys
from pathlib import Path

# Corrige o problema de WindowsPath no macOS
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath
```

### Ajuste dos Caminhos

Certifique-se de que o caminho do modelo e do vídeo estejam corretos:

```python
model_path = "yolov5/runs/train/twenty-epochs/weights/best.pt"
video_test_path = "videos_teste/video.mp4"
```

Se o vídeo ou o modelo estiverem em diretórios diferentes, ajuste os caminhos de acordo.

## Passo 6: Executar o Projeto

Com todas as configurações feitas, execute o seguinte comando:

```bash
python objectDetection.py
```

Se tudo estiver correto, o sistema iniciará a detecção de objetos e enviará alertas por e-mail quando objetos suspeitos forem identificados.

## Estrutura Final do Projeto

```
GRUPO1-HACKATON/
├── env/
├── videos_teste/
│   └── video.mp4
├── yolov5/
│   ├── models/
│   ├── runs/
│   │   └── train/
│   │       └── twenty-epochs/
│   │           └── weights/
│   │               └── best.pt
│   └── utils/
├── .env
├── objectDetection.py
├── combine_image_and_annotations.py
└── README.md
```

## Solução de Problemas

1. **Erro `ModuleNotFoundError: No module named 'utils'`**  
    Verifique e corrija os imports para: `from yolov5.utils import ...`

2. **Erro `FileNotFoundError: No such file or directory`**  
    Verifique se o caminho do modelo `best.pt` está correto.

3. **Erro `UnsupportedOperation: cannot instantiate 'WindowsPath' on your system`**  
    O código de correção do `WindowsPath` resolve o problema para macOS.

---

Este guia cobre todo o processo desde a configuração inicial até a execução bem-sucedida do projeto. Se houver dúvidas, verifique cada passo para garantir que todas as dependências e caminhos estejam corretos.
