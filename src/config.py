# -*- coding: utf-8 -*-
"""
Ficheiro de Configuração Centralizado para o Projeto de GANs (Versão Final Corrigida)

Adiciona a lógica automática para descobrir o número de classes do dataset.
"""
from pathlib import Path
from datetime import datetime
from typing import Union

# ==============================================================================
# 1. CONFIGURAÇÕES PRINCIPAIS (OS PONTOS A MUDAR)
# ==============================================================================
DATASET_NAME = "mnist"
# Defina como True se a sua pasta de dados tiver subpastas 'train' e 'test'
HAS_TRAIN_TEST_SPLIT = True


class Debug:
    """Configurações para acelerar testes e depuração."""

    SUBSET_SIZE: Union[int, None] = (
        200  # Defina um número para usar apenas uma fração do dataset, ou None para usar tudo
    )


class Model:
    """Parâmetros da arquitetura e das imagens."""

    LATENT_DIM = 100

    # Defina um número (ex: 28) ou 'original'.
    IMG_HEIGHT: Union[int, str] = "original"
    IMG_WIDTH: Union[int, str] = "original"

    # MODO DE COR: Escolha explicitamente 'grayscale' (1 canal) ou 'RGB' (3 canais).
    COLOR_MODE: str = "grayscale"  # Mude para 'RGB' se estiver a usar imagens coloridas

    # O número de canais é definido automaticamente. Não mude esta linha.
    CHANNELS = 1 if COLOR_MODE == "grayscale" else 3


class Training:
    """Hiperparâmetros para o treino da GAN."""

    EPOCHS = 50
    BATCH_SIZE = 256
    GENERATOR_LR = 0.0002
    DISCRIMINATOR_LR = 0.0002
    ADAM_BETA_1 = 0.5
    SAVE_INTERVAL = 20


# ==============================================================================
# 2. VARIÁVEIS AUTOMATIZADAS (NÃO PRECISA DE MUDAR)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / DATASET_NAME

# --- LÓGICA DE CONTAGEM DE CLASSES ADICIONADA ---
# Define de onde as classes devem ser contadas (da pasta 'train' ou da raiz do dataset)
if HAS_TRAIN_TEST_SPLIT:
    CLASS_COUNT_DIR = DATA_DIR / "train"
else:
    CLASS_COUNT_DIR = DATA_DIR

try:
    # Conta as subpastas no diretório para descobrir as classes
    CLASS_NAMES = sorted([d.name for d in CLASS_COUNT_DIR.iterdir() if d.is_dir()])
    NUM_CLASSES = len(CLASS_NAMES)
except FileNotFoundError:
    print(
        f"AVISO: Diretório '{CLASS_COUNT_DIR}' não encontrado para contar as classes."
    )
    CLASS_NAMES = []
    NUM_CLASSES = 0
# --- FIM DA LÓGICA DE CONTAGEM ---

PROJECT_NAME = f"{DATASET_NAME}_ACGAN"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


class Paths:
    """Organiza todos os caminhos de saída do projeto."""

    OUTPUT_DIR = BASE_DIR / "output"
    TMP_RUN_DIR = OUTPUT_DIR / f"_tmp_{PROJECT_NAME}_{TIMESTAMP}"
    GENERATED_IMAGES_DIR = TMP_RUN_DIR / "generated_images"
    MODELS_DIR = TMP_RUN_DIR / "models"
    GENERATOR_MODEL_FILE = MODELS_DIR / "generator.keras"


# ==============================================================================
# AÇÃO FINAL: Cria os diretórios necessários para a execução
# ==============================================================================
Paths.GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
Paths.MODELS_DIR.mkdir(exist_ok=True)
