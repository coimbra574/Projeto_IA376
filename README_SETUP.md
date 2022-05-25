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

```bash
make create_environment
# ou
conda env create -f environment.yml
```
### Instalação dos requirements
```bash
make requirements
# ou 
pip install -r requirements.txt
```

#### Atualização das dependências do projeto

Para adicionar uma nova dependência ao projeto, deve-se adicioná-la no arquivo `requirements.in`, que é utilizado para gerar o `.txt` com o comando

```bash
pip-compile requirements.in
```
