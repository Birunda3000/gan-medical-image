# -*- coding: utf-8 -*-
"""
Ficheiro de Configuração Centralizado para o Projeto de GANs.

Contém todos os parâmetros para o treino do Gerador e do Discriminador,
facilitando a experimentação e a gestão do projeto.
"""
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. CONFIGURAÇÕES PRINCIPAIS (OS PONTOS A MUDAR)
# ==============================================================================
DATASET_NAME = "mnist" # Ex: 'mnist', 'cifar10', 'covid_images'

class Debug:
    """Configurações para acelerar testes e depuração."""
    # Defina um número para usar apenas N imagens por classe (ou do dataset total).
    # Defina como None para usar o dataset completo.
    SUBSET_SIZE = 2000 # Usar apenas 2000 imagens para um teste rápido

class Model:
    """Parâmetros da arquitetura dos modelos."""
    # O Gerador usa o espaço latente como entrada para criar imagens.
    LATENT_DIM = 100
    # Parâmetros das imagens (saída do Gerador, entrada do Discriminador)
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1 # 1 para escala de cinza (MNIST), 3 para RGB

class Training:
    """Hiperparâmetros para o processo de treino da GAN."""
    EPOCHS = 50
    BATCH_SIZE = 256
    
    # Taxas de aprendizagem separadas para o Gerador e o Discriminador
    GENERATOR_LR = 0.0002
    DISCRIMINATOR_LR = 0.0002
    
    # Parâmetro Beta1 para o otimizador Adam, comum em GANs
    ADAM_BETA_1 = 0.5
    
    # Frequência para salvar imagens de amostra e o modelo
    SAVE_INTERVAL = 5 # Salvar a cada 5 épocas

# ==============================================================================
# 2. VARIÁVEIS AUTOMATIZADAS (NÃO PRECISA DE MUDAR)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / DATASET_NAME
PROJECT_NAME = f"{DATASET_NAME}_DCGAN"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

class Paths:
    """Organiza todos os caminhos de saída do projeto."""
    OUTPUT_DIR = BASE_DIR / "output"
    # Pasta temporária para a execução atual
    TMP_RUN_DIR = OUTPUT_DIR / f"_tmp_{PROJECT_NAME}_{TIMESTAMP}"
    
    # Subpastas dentro da execução para melhor organização
    GENERATED_IMAGES_DIR = TMP_RUN_DIR / "generated_images"
    MODELS_DIR = TMP_RUN_DIR / "models"
    
    # Caminho para salvar o modelo do gerador
    GENERATOR_MODEL_FILE = MODELS_DIR / "generator.h5"

# ==============================================================================
# AÇÃO FINAL: Cria os diretórios necessários para a execução
# ==============================================================================
Paths.GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
Paths.MODELS_DIR.mkdir(exist_ok=True)