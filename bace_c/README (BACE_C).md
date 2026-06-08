```
## ⚙️ Dependências e Configuração do Ambiente

Este projeto foi desenvolvido e testado utilizando **Python 3.10.6**. 
Para garantir a reprodutibilidade dos resultados e evitar problemas de 
compatibilidade (especialmente com a biblioteca Keras/TensorFlow), 
recomenda-se a utilização das versões exatas listadas abaixo.

### 1. Instalação de Pacotes (CPU / Standard)

Para instalar todas as dependências necessárias para executar os 
notebooks, abra o terminal na pasta do projeto e execute o seguinte 
comando:

# 1. Criar o ambiente virtual
python -m venv qsar_env

# 2. Ativar o ambiente (Windows)
qsar_env\Scripts\activate

# 3. Instalar as dependências do projeto
pip install tensorflow==2.10.0 pandas numpy scikit-learn matplotlib seaborn jupyter

(Nota: O tensorflow==2.10.0 é compatível de forma nativa com a CPU. 
Se o ambiente não possuir uma placa gráfica configurada, o código será 
executado normalmente utilizando o processador).

### 2. Configuração com Aceleração por GPU (Opcional)

Para tempos de treinamento reduzidos, o projeto foi validado em um 
ambiente com aceleração por hardware. O TensorFlow 2.10 é a última 
versão com suporte nativo para GPU em sistemas Windows. Se desejar 
executar o código com a GPU, o seu sistema deve possuir a seguinte 
stack da NVIDIA devidamente instalada e mapeada nas variáveis de ambiente:

* NVIDIA GPU Driver (compatível)
* CUDA Toolkit: 11.2
* cuDNN: 8.1.0

Você pode verificar se a GPU está sendo reconhecida pelo TensorFlow 
executando a primeira célula do notebook, que deverá retornar: 
"GPU Disponível: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]".

### 3. Estrutura de Diretórios dos Dados

Para que os arquivos .ipynb encontrem as imagens das moléculas 
corretamente, certifique-se de que as pastas dos conjuntos de dados 
extraídos se encontram na raiz do projeto, respeitando a seguinte estrutura:

📁 raiz_do_projeto/
 ├── 📄 1 BACE_C.ipynb
 ├── 📁 dataset_bace_c/
 │    ├── 📁 train/
 │    ├── 📁 val/
 │    └── 📁 test/

### 4. Execução e Logs Automáticos

Basta abrir os arquivos .ipynb em um ambiente Jupyter (via navegador 
ou VS Code) e executar "Restart Kernel and Run All Cells". 

Os algoritmos estão programados para:
1. Treinar a arquitetura CNN construída do zero (Baseline) e a rede com 
   transferência de aprendizado (MobileNetV2).
2. Criar automaticamente uma pasta chamada "resultados_BACE_C" 
   (ou "resultados_BBBP").
3. Exportar para esta pasta os gráficos de treino, o relatório de 
   métricas em .txt e a tabela consolidada em .csv ao finalizar a execução.
```
