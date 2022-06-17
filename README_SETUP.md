# Setup

A estrutura básica do repositório foi criada com o cookie cutter para data science [deste link](https://github.com/drivendata/cookiecutter-data-science).

## Clone
Esse repositório contém submódulos. Para fazer o pull completo:

```bash
git clone --recurse-submodules https://github.com/coimbra574/Projeto_IA376
```

Ou, após já clonado:

```bash
git submodule init
git submodule update
```

Outros comandos úteis na [documentação de submódulos do git](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

O submódulo principal do projeto é este repositório: https://github.com/luizapozzobon/denoising-diffusion-gan. Branch: ia376


## Criação do ambiente de desenvolvimento

Para criar e ativar o ambiente conda, instalar os requirements e os módulos do projeto:

<details>
  <summary>Passo a passo</summary>

    ```bash
    make create_environment
    # ou
    conda env create -f environment.yml
    ```

    ```
    conda activate ia376-diffusion
    ```

    ```bash
    make requirements
    ```

    Para uso de scripts dos módulos do projeto `src` e dos submódulos adicionados com o git, é necessário instalá-los em modo editável. O comando abaixo faz o que é necessário:

    ```bash
    make setup
    ```

    Isso permite o acesso a todos aos submódulos do git e os pacotes são atualizados de acordo com as modificações nos arquivos .py.

    Fonte: https://stackoverflow.com/questions/6323860/sibling-package-imports

</details>

### Atualização das dependências do projeto

Para adicionar uma nova dependência ao projeto, deve-se adicioná-la no arquivo `requirements.in`, que é utilizado para gerar o `.txt` com o comando

```bash
pip-compile requirements.in
```

## Download dos pesos dos modelos

Os pesos estão disponíveis no [drive do projeto](https://drive.google.com/drive/folders/1_1NgQeAKMasldrXMclZT_M4AzfjP8Jbl?usp=sharing). Baixe a pasta `models` inteira e a coloque na raiz do repositório (`Projeto_IA376/models`).

## Geração de resultados

### Geração de amostras de cada modelo

```bash
make sample
```

Ou individualmente:

```bash
make sample_original
make sample_ddgan
```

### Geração das distribuições amostradas pelos modelos

```bash
make distributions
```
