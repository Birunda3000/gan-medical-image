# Use a imagem base da NVIDIA TensorFlow que você puxou
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Instala as bibliotecas de dependência gráfica necessárias para o OpenCV
# --no-install-recommends: evita instalar pacotes recomendados desnecessários
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

# Defina o diretório de trabalho padrão dentro do container para o seu projeto
WORKDIR /tf/TCC_GAN

# Copie o arquivo requirements.txt para o diretório de trabalho do container
COPY requirements.txt .

# Instale as dependências Python listadas em requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão para iniciar o Jupyter Lab
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0"]