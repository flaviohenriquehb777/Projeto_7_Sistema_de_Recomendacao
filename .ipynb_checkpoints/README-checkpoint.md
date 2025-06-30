# Sistema de Recomendações (Sam's Club - Walmart)

**Sistema de recomendações de produtos para consumidores do Sam's Club - Walmart**

## Sumário
- [Descrição](#descrição)
- [Motivação](#motivação)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Instalação e Uso](#instalação-e-uso)
- [Resultados e Recomendações](#resultados-e-recomendações)
- [Observações](#observações)
- [Licença](#licença)
- [Contato](#contato)

## DESCRIÇÃO:

Este modelo implementa um sistema de recomendação baseado em técnicas de Deep Learning, utilizando TensorFlow (Keras) para treinar um modelo com uma base de dados. O sistema gera, ao final, uma lista dos 07 (sete) melhores produtos recomendados para o consumidor.

## Motivação:

O objetivo deste projeto é aprimorar a experiência de compra dos clientes do Sam's Club e Walmart, oferecendo recomendações de produtos personalizadas. Isso visa aumentar a satisfação do cliente e otimizar as vendas através de sugestões mais relevantes e direcionadas.

## ALGUMAS TECNOLOGIAS UTILIZADAS:
- Python
- Pandas
- Numpy
- TensorFlow (Keras)
- SHAP

## Estrutura do Projeto:
- `data/`: Contém os dados brutos e o arquivo `data.parquet` após o pré-processamento.
- `notebooks/`: Inclui os notebooks Jupyter para Análise Exploratória de Dados (EDA) e treinamento do modelo.
    - `01_777_Sistema_Recomendacao_Inicial.ipynb`: Notebook para a análise exploratória de dados e pré-processamento inicial.
    - `02_777_Sistema_Recomendacao_Final.ipynb`: Notebook principal com o desenvolvimento e treinamento do modelo de recomendação.
    - `03_777_Sistema_Recomendacao_Producao.ipynb`: Notebook com a versão do modelo focada em produção.
    - `04_777_Metodo_SHAP.ipynb`: Notebook que explora a interpretabilidade do modelo utilizando o método SHAP.
- `README.md`: Este arquivo.
- `LICENSE.md`: Arquivo contendo a licença do projeto (MIT).

## Processo de Desenvolvimento:

No arquivo inicial (`01_777_Sistema_Recomendacao_Inicial.ipynb`), foi feita uma análise exploratória e, com isso, foram realizadas algumas correções na base de dados:
- Eliminação dos valores nulos;
- Correção de alguns tipos de dados (types);
- Foi feito o downcast dos dados para otimização de memória;
- Ao final, os dados tratados foram salvos na extensão parquet.

O arquivo final (`02_777_Sistema_Recomendacao_Final.ipynb`) inicia importando a base no formato parquet e os passos que se seguem para a construção do modelo são os seguintes:
- Filtrar colunas relevantes;
- Codificar os nomes dos clientes e IDs dos produtos;
- Normalizar as vendas (utilizando o MinMaxScaler);
- Criar um conjunto de dados TensorFlow (`tf.data.Dataset`);
- Definir dimensões dos embeddings para clientes e produtos;
- Criar as camadas de embeddings para representar clientes, produtos, categorias e subcategorias;
- Definição da arquitetura do modelo de Deep Learning;
- Concatenar todas as informações para alimentar o modelo;
- Adicionar camadas densas para aprender padrões complexos nas interações;
- Treinar o modelo utilizando os dados preparados;
- Obter o ID da cliente "Irene Maddox" (utilizada como exemplo para o modelo criar recomendações);
- Criar recomendações para Irene;
- Obter os 7 melhores produtos recomendados;
- Criar um DataFrame para exibir as recomendações de forma clara;
- Implementar uma função genérica (`recomendar_produtos`) para determinar as 7 melhores recomendações para qualquer cliente escolhido;
- Utilizar a função `recomendar_produtos` com o cliente "Darrin Van Huff" como exemplo para demonstrar sua aplicabilidade.

## Instalação e Uso:

Para configurar e rodar o projeto localmente, siga os passos abaixo:

1.  **Pré-requisitos:**
    * Python 3.8+
    * pip

2.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/Projeto_7_Sistema_de_Recomendacao.git](https://github.com/seu-usuario/Projeto_7_Sistema_de_Recomendacao.git)
    cd Projeto_7_Sistema_de_Recomendacao
    ```
    (Certifique-se de substituir `seu-usuario` pelo seu nome de usuário do GitHub.)

3.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

4.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    **(Importante: Crie o arquivo `requirements.txt` executando `pip freeze > requirements.txt` no seu ambiente virtual após instalar todas as bibliotecas usadas nos notebooks.)**

5.  **Como usar:**
    Para explorar o projeto e gerar recomendações, você pode abrir e executar os notebooks Jupyter na pasta `notebooks/`. O notebook `02_777_Sistema_Recomendacao_Final.ipynb` contém a lógica principal de recomendação.

    Você pode chamar a função `recomendar_produtos` (após carregá-la do notebook ou de um script):

    ```python
    # Exemplo de uso da função recomendar_produtos (após carregar o modelo e encoders)
    # Assumindo que você já carregou o modelo, customer_encoder, product_encoder, etc.
    # Essas etapas são detalhadas no notebook 02_777_Sistema_Recomendacao_Final.ipynb

    # Exemplo para um cliente existente na base
    recomendacoes_darrin = recomendar_produtos("Darrin Van Huff")
    print("Recomendações para Darrin Van Huff:")
    print(recomendacoes_darrin)

    # Exemplo para um novo cliente (conforme a 'OBSERVAÇÃO' abaixo)
    recomendacoes_novo = recomendar_produtos("Novo Cliente Qualquer")
    print("\nRecomendações para um Novo Cliente Qualquer:")
    print(recomendacoes_novo)
    ```

## Resultados e Recomendações:

No exemplo apresentado, os 07 (sete) produtos recomendados são para o cliente 'Darrin Van Huff'. Abaixo, uma captura de tela do DataFrame de recomendações gerado pelo sistema:

![Screenshot da Saída do Modelo](img/Screenshot_saida_modelo.png)

## OBSERVAÇÕES:

A função 'recomendar_produtos', desenvolvida por mim e presente no notebook final, possibilita que novos clientes, ou seja, aqueles que não integram a base de dados original, também recebam recomendações dos 07 (sete) melhores produtos. Isso é crucial para a escalabilidade e aplicabilidade do sistema em cenários reais de novos usuários.

## Licença:

Este projeto está licenciado sob a Licença MIT. Para mais detalhes, consulte o arquivo [LICENSE.md](LICENSE.md) na raiz do repositório.

## Contato:

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
- **Nome:** Flávio Henrique Barbosa
- **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
- **Email:** flaviohenriquehb777@outlook.com
