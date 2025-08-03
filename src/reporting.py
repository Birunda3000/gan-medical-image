# (em src/reporting.py)
# -*- coding: utf-8 -*-
"""
Módulo para a geração de relatórios de texto pós-treino para a GAN.
"""
from pathlib import Path
from typing import Dict, Any

try:
    from . import config
except ImportError:
    import config

def save_gan_report(
    report_path: Path,
    timings: Dict[str, Any],
    num_images: int,
    final_losses: Dict[str, float]
):
    """Gera e escreve o relatório final para uma execução de treino da GAN."""
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("      RELATÓRIO FINAL DA EXECUÇÃO DE TREINO (ACGAN)\n")
            f.write("="*60 + "\n\n")

            f.write(f"Projeto: {config.PROJECT_NAME}\n")
            f.write(f"Dataset: {config.DATASET_NAME}\n")
            f.write(f"Timestamp: {config.TIMESTAMP}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Resumo dos Tempos\n")
            f.write("-" * 60 + "\n")
            f.write(f"Início: {timings['start_dt']}\n")
            f.write(f"Fim:    {timings['end_dt']}\n")
            f.write(f"Duração Total: {(timings['total_duration'])/60:.2f} minutos\n\n")

            f.write("-" * 60 + "\n")
            f.write("Parâmetros da Execução (config.py)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Épocas: {config.Training.EPOCHS}\n")
            f.write(f"Tamanho do Lote (Batch Size): {config.Training.BATCH_SIZE}\n")
            f.write(f"Dimensão Latente: {config.Model.LATENT_DIM}\n")
            f.write(f"Taxa de Aprendizagem (Gerador): {config.Training.GENERATOR_LR}\n")
            f.write(f"Taxa de Aprendizagem (Discriminador): {config.Training.DISCRIMINATOR_LR}\n")
            f.write(f"Número de Imagens de Treino: {num_images}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Métricas Finais (Última Época)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Perda do Gerador: {final_losses['g_loss']:.4f}\n")
            f.write(f"Perda do Discriminador: {final_losses['d_loss']:.4f}\n")

        print(f"✅ Relatório da execução salvo em: {report_path}")
    except Exception as e:
        print(f"Erro ao escrever o relatório: {e}")