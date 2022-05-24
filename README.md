# `Sobre fairness na geração não condicional com modelos de difusão`
# `On the fairness of unconditional generation with Diffusion Models`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*,
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização |
> |--|--|--|
> | Álvaro Airemoraes Capelo | 104534  | Eng. de Computação (DCA) |
> | Luiza Amador Pozzobon  | 233818  | Eng. de Computação (DCA) |
> | Tainá de Souza Coimbra  | 157305  | Eng. Elétrica/Computação (DECOM)|


## Resumo

A análise da capacidade de generalização e da representatividade em modelos generativos, em particular sobre a ótica de vieses, tem ganhado bastante interesse da comunidade científica desde o recente renovado interesse na área de síntese de imagens proporcionado pelo desenvolvimento das Redes Generativas Adversárias. Tem-se observado que essa família de arquiteturas produz viéses de amostragem, o que tem despertado intenso interesse da comunidade científica. Recentemente, um modelo estado-da-arte foi proposto[11], e os resultados reportados apontam para a solução do problema de representatividade (medido pela presença de todas as modas) do conjunto de dados de treino das GANs.

Nesse trabalho, será avaliado se essa cobertura de modas significa também paridade demográfica [4]. Serão treinados um modelo dessa arquitetura estado-da-arte, a _Denoising Diffusion GAN_, e um modelo da família das GANs, a ser definido, em um conjunto de dados MNIST modificado [5]. Dois grupos serão criados no conjunto de dados: i) número branco e fundo branco, e ii) número preto e fundo branco. Serão usadas três proporções para cada grupo, 30:70, 50:50 e 70:30, e amostras geradas de forma não condicionada terão as densidades de probabilidades de cada grupo medidas e comparadas com a do conjunto de dados de treino.

## Descrição do Problema e Motivação

São frequentes os relatos de bias na geração de imagens sintéticas com Generative Adversarial Networks [1]. Um exemplo que repercutiu na mídia foi o caso em que o algoritmo PULSE [2], que faz o upsample de imagens em baixa resolucção, [transformou o presidente Barack Obama em um homem branco](https://twitter.com/Chicken3gg/status/1274314622447820801). Esse algoritmo utiliza uma StyleGAN [3] como base do espaço de busca para o upsampling.

![PULSE bias](https://pbs.twimg.com/media/Ea9GZUbXsAEORIb?format=jpg&name=small)

Figura 1: Algoritmo PULSE [2] faz o upsampling do presidente Barack Obama para um homem branco.

Os resultados de um estudo posterior indicaram biases implícitos na StyleGAN [4] e bias racial nas imagens geradas: cerca de 73% das imagens geradas pelo método são de pessoas brancas, enquanto pessoas asiáticas e pretas aparecem em cerca de 14 e 10% das imagens sintéticas, respectivamente. Conforme os autores do estudo, a não geracão de faces sintéticas de grupos sub-representados diminui ainda mais a habilidade desses indivíduos de serem vistos e ouvidos [4]. Outro experimento com um conjunto de dados controlado [5] observou que, mesmo com uma população de treino constituída de classes balanceadas, StackedGANs [6] são incapazes de gerar imagens sintéticas com distribuições equivalentes. O problema é intensificado quando o conjunto de treino  ́e desbalanceado. Conclui-se, portanto, que o uso dessa família de modelos pode enviesar fortemente as tarefas posteriores para as quais os dados sintéticos serão utilizados, já que não respeitam o conceito de *statistical parity* onde “a demografia do conjunto de indivíduos recebendo qualquer classificação é equivalente à demografia da população como um todo” [7].

Recentemente outro modelo generativo obteve repercussão devido à qualidade das imagens sintéticas: o DALLE-2 [8]. O resultado foi atingido a partir do poder conjunto de modelos de difusão [9] e de *embeddings* do CLIP [10]. Neste trabalho, objetiva-se o estudo dos possíveis biases presentes em modelos de difusão, de forma similar ao que foi realizado por [5] para as StackedGANs. Mais especificamente, serão comparadas as distribuicões das imagens geradas versus as distribuicões do conjunto de treino controlado para Denoising Diffusion GANs (DDGANs) [11] e uma arquitetura de GAN que ainda será definida.

## Objetivos

Avaliar se a solução proposta pela arquitetura _Desnoising Diffusion GANs_ (DDGANs) para o problema de cobertura de moda das arquiteturas baseadas em GANs também significa que paridade demográfica [4] é atingida na distribuição das imagens geradas.

### Objetivos específicos

Avaliar se a densidade de probabilidade para os diferentes grupos de imagens do conjunto de dados de treino é reproduzida pelos modelos DDGAN e a rede GAN a ser definida.

## Metodologia Proposta

Neste projeto investigaremos a capacidade de adequação à métrica de paridade demográfica [4] de modelos generativos, com ênfase na comparação de modelos de difusão [9] com GANs [1]. Para isso, serão elaborados *toy problems* com o dataset MNIST para avaliação da distribuição das imagens sintéticas quando comparadas à distribuição do conjunto de treino. O modelo de difusão estudado será o _Denoising Diffusion GAN_ (DDGAN) [11] e a arquitetura de GAN ainda será definida.

### Conjunto de dados

O conjunto de dados MNIST sera modificado para todas as classes (dígitos) venham de dois grupos:
- Grupo 1: imagens tradicionais do MNIST, dígitos em branco, fundo preto
- Grupo 2: imagens MNIST invetidas, com dígitos em preto e fundo branco

Entao, produziremos três cenários de avaliação, variando as proporções de cada grupo:
- Cenario A: Grupo 1 e 2 com 30 e 70% do conjunto de treino, respectivamente.
- Cenario B: Grupos 1 e 2 com 50 e 50% do conjunto de treino, respectivamente.
- Cenario C: Grupos 1 e 2 com 70 e 30% do conjunto de treino, respectivamente.

### Abordagens de modelagem generativa

#### DDGAN

*Denoising Diffusion GAN* é a rede proposta por [11] para combater o `trilemma` dos modelos generativos usuais (GANs, VAEs e modelos de difusão), cujos resultados são sempre um `trade-off` entre três fatores: (1) rápida amostragem, (2) alta qualidade e (3) cobertura das modas dos dados de treino. Conforme a Figura 2, modelos de difusão [9] geram amostras de ótima qualidade e com cobertura das modas do conjunto de treino, mas o tempo de amostragem é elevado, impedindo seu uso em aplicações do mundo real. Na arquitetura proposta, o processo de `denoising` é modelado por uma *Conditional GAN* [12], em vez de por Gaussianas Multimodais como nos modelos de difusão clássicos. Essa mudança na arquitetura, ilustrada na Figura 3, permite aumentar em até 2000x a velocidade de amostragem para o conjunto CIFAR10 mantendo a qualidade dos dados sintéticos do modelo de difusão original. A arquitetura da DDGAN pode ser observada na Figura 4.

![The generative trilemma](images/trilemma.png)
Figura 2. O _trilemma_ de modelos genrativos

![Denoising with Unimodal Gaussian vs GANS](images/diffusion_gaussian_vs_gans.png)
Figura 3. Mudança introduzida pelas DDGANs: modelagem de _denoising_ usando GANs multimodais

![DDGAN](images/ddgan.png)
Figura 4. Arquitetura da DDGAN

#### Alguma arquitetura de GAN

### Ferramentas a serem utilizadas

- Bibliotecas padrão de data science (Python 3, PyTorch, Pandas, etc.)
- GitHub
- Google Colab
- Kaggle
- Overleaf

### Proposta de avaliação

Propomos utilizar abordagem semelhante ao que foi feito em[5], isto é, medir a densidade de probabilidade dos grupos nas imagens sintetizadas de forma não condicionada e comparar com as densidades nos conjuntos de dados de treino. Com esta metodologia se propõe avaliar a representatividade e o viés em imagens sintéticas aproximando o conceito de paridade demográfica usado por Salminem e colaboradores, em que um algoritmo sem viés deveria produzir imagens para cada grupo com a mesma probabilidade [4].

### Resultados esperados

Até o momento, foi feita a readequação da proposta do projeto, visto que houve uma mudança temática. Foi decidido trabalhar com o artigo [11], mais especificamente o problema de viés, questão não muito explorada pelo autor, que focou apenas na demonstração de que a DDGAN cobre todas as modas. Isto é diferente da proposta do nosso trabalho, que pretende avaliar como diferentes proporções de grupos no conjunto de dados de treino afetam a paridade demográfica desses grupos no resultado. A fim de explorar melhor este assunto, prosseguimos com estudos de artigos relativos a modelos de difusão, problemas de viés e como medi-los, cobertura de moda, etc.

Em termos práticos, foi feita a configuração do código da DDGAN apresentado em [11] para rodar a base MNIST e o MNIST controlado (adaptado com diferentes proporções). O código da DDGAN é extenso e a estrutura utilizada para treino é complexa, dificultando ligeiramente a manipulação. Entretanto, são fornecidos diversos parâmetros de entrada que permitem modificar relativamente a estrutura principal, assim possibilitando o uso no Google Colab comum. 

### Conclusão

Será analisada a paridade demográfica [4] dos modos em modelos de difusão, em especial a DDGAN que utiliza uma GAN condicional durante o processo de denoising. Pelo trilema das redes generativas [11], modelos de difusão são eficientes para gerar amostras de alta qualidade e diversidade (baixo viés), porém com um alto tempo de amostragem. A ideia da GAN para denoising é justamente diminuir este tempo de amostragem. Queremos verificar o quanto o uso da GAN na DDGAN afeta o viés ao mesmo tempo em que mantém este baixo tempo de amostragem, visto que a GAN é conhecida por produzir amostras com pouca diversidade. Para identificar essa influência do viés nos modelos, vamos utilizar como base as métricas descritas em [5]. Já começamos a modificar o código da DDGAN para a nossa base MNIST modificada, mas ainda faltam algumas etapas para a finalização do projeto. Para isso, precisamos: (i) Terminar de ajustar o código e treinar a DDGAN para a base MNIST modificada (ii) Rodar resultados da DDGAN para datasets de diferentes proporções do MNIST controlado, (iii) Implementação de uma GAN para comparações de resultados com a DDGAN, (iv) Análise dos resultados com base em métricas de viés, (v) Escrita de relatório. Estas etapas estão sumarizadas no cronograma abaixo.


## Cronograma

| Tarefa                                               | 20/04 | 27/04 | 04/05 | 11/05 | 18/05 | 25/05 | 01/06 | 08/06 | 15/06 | 22/06 | 29/06 | 04/07 | 06/07 |
|------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Readequação da proposta e levantamento bibliográfico | x     | x     | x     | x     | x     | x     |       |       |       |       |       |       |       |
| Experimentações iniciais com DDGAN                   |       |       |       | x     | x     | x     |       |       |       |       |       |       |       |
| Desenvolvimento dos cenários de experimentação       |       |       |       |       | x     | x     | x     |       |       |       |       |       |       |
| E2 - Checkpoint                                      |       |       |       |       |       | x     |       |       |       |       |       |       |       |
| Alterações e validação do código da DDGAN            |       |       |       |       |       | x     | x     | x     |       |       |       |       |       |
| Treinamento da DDGAN                                 |       |       |       |       |       |       | x     | x     | x     | x     | x     |       |       |
| Treinamento da GAN                                   |       |       |       |       |       |       | x     | x     | x     | x     | x     |       |       |
| E3 - Entrega do código                               |       |       |       |       |       |       |       |       |       |       |       | x     |       |
| Escrita de relatório e compilação de resultados      |       |       |       |       |       |       | x     | x     | x     | x     | x     | x     | x     |

## Referências Bibliográficas

[1] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.

[2] S. Menon, A. Damian, S. Hu, N. Ravi, and C. Rudin, “Pulse: Self-supervised photo upsampling via latent space exploration of generative models,” in Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 2020, pp. 2437–2445.

[3] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4401–4410.

[4] J. Salminen, S.-g. Jung, S. Chowdhury, and B. J. Jansen, “Analyzing demographic bias in artificially generated facial pictures,” in Extended Abstracts of the 2020 CHI Conference on Human Factors in Computing Systems, 2020, pp. 1–8.

[5] P. J. Kenfack, D. D. Arapov, R. Hussain, S. A. Kazmi, and A. Khan, “On the fairness of generative adversarial networks (gans),” in 2021 International Conference” Nonlinearity, Information and Robotics”(NIR). IEEE, 2021, pp. 1–7.

[6] X. Huang, Y. Li, O. Poursaeed, J. Hopcroft, and S. Belongie, “Stacked generative adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5077–5086.

[7] C. Dwork, M. Hardt, T. Pitassi, O. Reingold, and R. Zemel, “Fairness through awareness,” in Proceedings of the 3rd innovations in theoretical computer science conference, 2012, pp. 214–226.

[8] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022.

[9] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840–6851, 2020.

[10] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.

[11] Z. Xiao, K. Kreis, and A. Vahdat, “Tackling the generative learning trilemma with denoising diffusion gans,” arXiv preprint arXiv:2112.07804, 2021.

[12] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv preprint arXiv:1411.1784, 2014.
