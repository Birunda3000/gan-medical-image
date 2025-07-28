# -*- coding: utf-8 -*-
"""
Módulo para criação de pipelines de dados para GANs (Versão Final Corrigida)

Funcionalidades:
- Carregamento de subconjunto de dados.
- Redimensionamento proporcional CORRIGIDO.
- Modo de cor flexível com avisos.
- Cache para processamento único de dados.
"""
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Union, List
import sys

try:
    from . import config
except ImportError:
    import config

Dataset = tf.data.Dataset

def _print_warning(path, original_channels, target_channels):
    """Função Python para imprimir um aviso de conversão de cor."""
    path_str = path.numpy().decode('utf-8')
    original_channels_int = original_channels.numpy()
    message = (f"\n\033[93m" f"[ AVISO DE CONVERSÃO DE COR ]\n" f"IMAGEM: {path_str}\n" f"PROBLEMA: A imagem original tem {original_channels_int} canal(is), mas o config.py forçou a leitura para {target_channels} canais.\n" f"AÇÃO: O TensorFlow fará a conversão automaticamente.\033[0m")
    print(message, file=sys.stderr)
    return tf.constant(True)

def _get_file_paths_and_labels(data_path: Path, subset_size: Union[int, None]) -> Tuple[List[str], List[int]]:
    """Função interna para listar os caminhos dos ficheiros e os seus rótulos a partir das subpastas."""
    image_paths = []
    image_labels = []
    class_name_to_id = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    
    print(f"A procurar por imagens em: {data_path}")
    if subset_size:
        print(f"Modo Debug: A usar um subconjunto de {subset_size} imagens no total.")

    for class_dir in data_path.iterdir():
        if not class_dir.is_dir() or class_dir.name not in class_name_to_id:
            continue
        class_name = class_dir.name
        label_id = class_name_to_id[class_name]
        for image_file in sorted(class_dir.glob('*.*')):
            image_paths.append(str(image_file))
            image_labels.append(label_id)

    if subset_size and len(image_paths) > subset_size:
        indices = tf.range(start=0, limit=len(image_paths), dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)[:subset_size]
        image_paths = [image_paths[i] for i in shuffled_indices.numpy()]
        image_labels = [image_labels[i] for i in shuffled_indices.numpy()]

    return image_paths, image_labels

def create_acgan_dataset(data_path: Path, batch_size: int) -> Tuple[Dataset, int]:
    """Cria um pipeline tf.data completo para o treino da ACGAN, retornando (imagem, rótulo)."""
    print(f"A carregar imagens de: {data_path} com COLOR_MODE='{config.Model.COLOR_MODE}'")
    
    paths, labels = _get_file_paths_and_labels(data_path, config.Debug.SUBSET_SIZE)
    if not paths:
        raise ValueError(f"Nenhuma imagem encontrada em {data_path}.")
        
    num_images = len(paths)
    print(f"Encontradas {num_images} imagens em {config.NUM_CLASSES} classes.")
    
    path_ds = Dataset.from_tensor_slices((paths, labels))
    
    def _parse_and_process_image(path, label):
        """Função interna que agora também recebe e retorna o rótulo."""
        img_raw = tf.io.read_file(path)
        
        if config.Model.COLOR_MODE != 'auto':
            original_shape = tf.io.image.decode_image(img_raw, expand_animations=False).shape
            original_channels = original_shape[-1] if len(original_shape) == 3 else 1
            target_channels = config.Model.CHANNELS
            if original_channels != target_channels:
                tf.py_function(_print_warning, inp=[path, original_channels, target_channels], Tout=tf.bool)

        img = tf.io.decode_image(img_raw, channels=config.Model.CHANNELS, expand_animations=False)
        
        # --- LÓGICA DE REDIMENSIONAMENTO CORRIGIDA ---
        if isinstance(config.Model.IMG_HEIGHT, int) and isinstance(config.Model.IMG_WIDTH, int):
            img = tf.image.resize(img, [config.Model.IMG_HEIGHT, config.Model.IMG_WIDTH])
        
        elif config.Model.IMG_HEIGHT == 'original' or config.Model.IMG_WIDTH == 'original':
            original_shape = tf.shape(img)
            original_height = tf.cast(original_shape[0], tf.float32)
            original_width = tf.cast(original_shape[1], tf.float32)

            if isinstance(config.Model.IMG_HEIGHT, int):
                new_height = tf.constant(config.Model.IMG_HEIGHT, dtype=tf.float32)
                aspect_ratio = original_width / original_height
                new_width = tf.cast(new_height * aspect_ratio, tf.int32)
                img = tf.image.resize(img, [tf.cast(new_height, tf.int32), new_width])
            
            elif isinstance(config.Model.IMG_WIDTH, int):
                new_width = tf.constant(config.Model.IMG_WIDTH, dtype=tf.float32)
                aspect_ratio = original_height / original_width
                new_height = tf.cast(new_width * aspect_ratio, tf.int32)
                img = tf.image.resize(img, [new_height, tf.cast(new_width, tf.int32)])

        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5
        return img, label

    dataset = (
        path_ds
        .shuffle(buffer_size=num_images)
        .map(_parse_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    print("Pipeline de dados avançado para a ACGAN criado com sucesso!")
    return dataset, num_images