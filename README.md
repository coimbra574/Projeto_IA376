# `Síntese Condicional de Tatuagens`
# `Conditional Tattoo Synthesis`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*,
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização |
> |--|--|--|
> | Álvaro Airemoraes Capelo | 104534  | Eng. de Computação (DCA) |
> | Luiza Amador Pozzobon  | 233818  | Eng. de Computação (DCA) |
> | Tainá de Souza Coimbra  | 157305  | Eng. Elétrica/Computação (DECOM)|


## Descrição Resumida do Projeto

Segundo o *Oxford Handbook of the Development of Imagination* "a imaginação é a capacidade de transcender mentalmente tempo, espaço e circunstâncias (Taylor 2013). Argúi-se que a *Imagination Science* será a área capaz de endereçar o problema de geração de amostras entendidas como "novas", "inovadoras" ou que se baseiam em conceitos criativos até então inacessíveis por pessoas. Arte, portanto, é um exemplo da capacidade imaginativa humana (Mahadevan, 2018).

O campo de *Deep Learning* experimenta na síntese de arte com técnicas como *Neural Style Transfer*, conforme a Figura 1, que baseia-se em redes convolucionais convencionais, como a *VGG*, e também com métodos baseados em redes generativas, como o *ArtGAN* e o *CAN* (*Creative Adversarial Network*). Os artigos de síntese de maior impacto baseiam-se em síntese de imagens naturais, como paisagens, ou de obras artísticas como quadros e pinturas. Objetiva-se entender como redes generativas se comportam em outros tipos de arte pouco exploradas, como tatuagens, que possuem características específicas do domínio e que diferem daquelas da arte convencional.

Nesse sentido, o objetivo principal do projeto é a síntese condicional de tatuagens de diferentes estilos, como em uma *CGAN* (Mirza e Osindero, 2014). 

[Link para o vídeo de apresentação da proposta.]()

![Neural Style Transfer](https://github.com/coimbra574/Projeto_IA376/blob/main/images/neural_style_transfer.jpeg)

## Metodologia Proposta

Para a primeira entrega, a metodologia proposta deve esclarecer:

### Base de dados

Como não encontramos nenhuma base viável para o projeto (número suficiente de imagens, etc.), decidimos fazer um “image scrapper” com dados do Instagram e Pinterest a partir do código fornecido pelo [artigo do medium](medium.com/vasily-betin/artificially-generated-tattoo-2d5fbe0f5146). Pretendemos coletar de 10000 a 20000 imagens para que a GAN consiga aprender a gerar tatuagens de cada estilo devidamente. Como são muitas imagens e precisamos ter a classe de cada uma de antemão para alimentar a GAN, pretendemos classificar manualmente cerca de 1500 imagens, e usar estas para treinar um classificador, assim identificando as imagens restantes. Finalmente, essa base de dados será então dividida em 3 partes: 

(i) - Uma para o treinamento da GAN

(ii) - Outra para o treino de classificadores para análise quantitativa dos resultados da GAN. Este classificador será alimentado com um dataset de diferentes proporções de dados reais e sintéticos

(iii) - Finalmente, uma proporção pequena de dados reais para validar os classificadores treinados anteriormente durante a análise

(Diagrama aqui)

### Abordagens de modelagem generativa

Nossa abordagem será usar uma Conditional GAN (CGAN), de forma que será passado como entrada também o label de cada imagem tanto para o network gerador quanto discriminativo. Esta estrutura é bem semelhante a GAN original, inclusive sua função de objetivo, porém as distribuições agora são condicionais:

(imagem função objetiva)
(imagem comparação estruturas GAN e CGAN)


### Ferramentas a serem utilizadas

- Bibliotecas padrão de data science (Python 3, PyTorch, Pandas, etc.)
- GitHub
- Programa para o tag de imagens
- Google Colab
- Overleaf

### Resultados esperados

Para a métrica FID, esperamos que o resultado seja baixo, mas até um certo limite para não apresentar overfitting. Já para o SSIM, quanto mais alto o valor, mais fiel a distribuição sintética será da distribuição real. Da mesma forma que o FID, também precisa ser um valor adequado para não termos overfitting. Durante a análise dos classificadores, é esperado que a métrica F1 seja muito parecida entre os resultados dos classifadores treinados com diferentes proporções de dados reais e sintéticos . Também esperamos que durante o MOS, os alunos tenham dificuldade de diferenciar tatuagens reais e sintéticas, e que consigam classificar facilmente entre os diferentes estilos. Vale ressaltar que existem limitações para o nosso modelo, já que a identificação inicial dos estilos não será feita por especialistas, e o número de imagens será limitado. 

### Proposta de avaliação

A avaliação quantitativa será feita com métodos de avaliação conhecidos, como Frechet Inception Distance (FID) e o Structural Similarity Index Measure (SSIM). O FID mede a "distância" entre as distribuições, enquanto o SSIM tenta medir a percepção de similaridade, procurando por características relevantes a percepção visual humana. Essas medidas foram escolhidas justamente por serem complementares: uma mede a fidelidade e outra a diversidade. Ainda dentro dos métodos quantitativos, também avaliaremos o desempenho de classificadores treinados com diferentes mixes de dados sintéticos e reais, que serão testados com os dados de teste definidos no item (iii). Por fim, utilizaremos o MOS (Mean Opinion Scores) como método qualitativo, usando como base os alunos da disciplina para diferenciar entre os diferentes estilos de tatuagem.


## Cronograma
| Tarefa  | 20/04 | 27/04 | 04/05 | 11/05 | 18/05 | 25/05 | 01/06 | 08/06 | 15/06 | 22/06 | 29/06 | 04/07 | 06/07 |
| ------- |:-------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Coleta e limpeza de dataset | X | X | X | X |
| Primeiras experimentações com arquitetura GAN | | X | X | X | X |
| E3 - Checkpoint | | | | | X |
| Aprofundamenteo de modelagem ||||| X | X | X | X |
| Avaliação |||||||| X | X | X | X |
| E4 - Entrega do código |||||||||||| X |
| Escrita de Relatório ||||||| X | X | X | X | X | X | X |

## Referências Bibliográficas

Mahadevan, Sridhar. "Imagination machines: A new challenge for artificial intelligence." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

Taylor, Marjorie. "1 Transcending Time, Place, and/or." The Oxford handbook of the development of imagination (2013): 3.

Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

Karras, Tero, et al. "Training generative adversarial networks with limited data." Advances in Neural Information Processing Systems 33 (2020): 12104-12114.

Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).

Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

Tan, Wei Ren, et al. "ArtGAN: Artwork synthesis with conditional categorical GANs." 2017 IEEE International Conference on Image Processing (ICIP). IEEE, 2017.

Elgammal, Ahmed, et al. "Can: Creative adversarial networks, generating" art" by learning about styles and deviating from style norms." arXiv preprint arXiv:1706.07068 (2017).
