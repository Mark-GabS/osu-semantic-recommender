# Guia de uso

Passo a passo para instalar, fazer buscas precisas e levar os mapas para o jogo.

- [1. Instalação](#1-instalação)
- [2. (Opcional) Conectar o Claude](#2-opcional-conectar-o-claude)
- [3. Primeira busca](#3-primeira-busca)
- [4. Como pedir de forma precisa](#4-como-pedir-de-forma-precisa)
- [5. Lendo o resultado](#5-lendo-o-resultado)
- [6. Baixar os mapas e criar a coleção](#6-baixar-os-mapas-e-criar-a-coleção)
- [7. Buscar só nos mapas que você já tem](#7-buscar-só-nos-mapas-que-você-já-tem)
- [8. Problemas comuns](#8-problemas-comuns)

---

## 1. Instalação

Requisitos: **Python 3.11+** e internet. Não é preciso ter conta no osu! nem chave de API.

**Linux / macOS**

```bash
git clone https://github.com/<usuario>/osu-semantic-recommender.git
cd osu-semantic-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/<usuario>/osu-semantic-recommender.git
cd osu-semantic-recommender
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sempre que abrir um terminal novo, entre na pasta e ative o ambiente (`source .venv/bin/activate` ou `.venv\Scripts\Activate.ps1`) antes dos comandos abaixo.

Para conferir que está tudo certo:

```bash
pytest
```

## 2. (Opcional) Conectar o Claude

Sem o Claude o programa funciona, mas entende os pedidos com regras fixas, que erram em frases livres como "5 estrelas e pouco" ou "nada muito streamado". Com o Claude, o pedido é interpretado por um modelo de linguagem.

O projeto usa o **Claude Code** instalado na sua máquina, com o login da sua conta Claude. Não há chave de API no projeto.

```bash
npm install -g @anthropic-ai/claude-code    # ou: curl -fsSL https://claude.ai/install.sh | bash
claude                                      # dentro dele: /login, depois /exit
python -m src.cli doctor                    # deve terminar com "ok — resposta estruturada"
```

O uso conta no seu plano do Claude. Se o `claude` não estiver no PATH, defina `CLAUDE_CODE_BIN` no `.env`.

## 3. Primeira busca

```bash
python -m src.cli recommend "quero mapas de 5.2 estrelas com jumps técnicos, rock japonês energético e poucas streams longas"
```

O programa busca no catálogo de mapas **ranked** do [osu.ppy.sh](https://osu.ppy.sh/beatmapsets), baixa o `.osu` dos melhores candidatos para medir streams, jumps e tech de verdade e mostra uma tabela com os links.

A primeira busca leva de 20 a 40 segundos (download dos `.osu`); buscas parecidas depois são mais rápidas por causa do cache.

## 4. Como pedir de forma precisa

O pedido vira duas coisas: **filtros rígidos** (números que o mapa precisa cumprir) e **preferências** (o que sobe no ranking).

### Números viram filtros rígidos

| Você escreve | O programa entende |
|---|---|
| `5.2 estrelas` / `5.2★` | entre 4.9 e 5.5 (±0.3) |
| `entre 5 e 6 estrelas` | 5.0 a 6.0 |
| `até 4 estrelas` | no máximo 4 |
| `acima de 7 estrelas` | no mínimo 7 |
| `200 bpm` | 195 a 205 |
| `AR 9.3` | 9.0 a 9.6 |
| `mapas curtos` / `mapas longos` | até 2 min / 3 min ou mais |
| `15 mapas` | devolve 15 (padrão: 20) |

**Regra de ouro: para ter precisão, dê números.** "Difícil" é vago; "6 estrelas" é exato.

### Mecânica vira preferência

| Palavra | O que busca |
|---|---|
| `jumps`, `aim`, `mira` | pulos |
| `técnico`, `tech` | ritmos misturados, SV variado, ângulos irregulares |
| `streams` | sequências de notas rápidas |
| `streams longas`, `deathstream` | streams enormes (stamina) |
| `bursts` | sequências curtas (3 a 6 notas) |
| `flow` | movimento contínuo e rápido |
| `AR alto`, `leitura` | AR alto |
| `sliders`, `SV` | mapas cheios de sliders |
| `velocidade`, `stamina` | velocidade de dedo / resistência |

### Negação

| Você escreve | Efeito |
|---|---|
| `sem streams` | remove mapas com stream maior que 6 notas |
| `poucas streams longas` | remove mapas com stream maior que 12 notas |
| `nada de jumps` | empurra mapas com jumps para baixo |

O limite de stream é conferido no arquivo `.osu` de cada mapa, algo que a busca do próprio site não consegue fazer.

### Gênero e clima

- **Gêneros:** rock japonês/jrock, rock, metal, j-pop, anime, vocaloid, eletrônica, clássica, piano, lofi, hip hop, jazz.
- **Climas:** energético, calmo, agressivo, alegre, emocional, elegante.

O gênero e o idioma vêm do próprio site do osu!, por isso "rock japonês" significa gênero Rock + idioma Japanese.

### Exemplos

| Vago | Preciso |
|---|---|
| "mapas difíceis de pulo" | "jumps entre 6 e 6.5 estrelas, sem streams" |
| "algo pra treinar" | "streams longas de 180 bpm para treinar stamina, até 5.5 estrelas" |
| "música legal" | "anime energético com bursts, 5 estrelas, mapas curtos" |
| "mapas calmos" | "flow calmo de piano, até 4 estrelas" |

### Conferir como o pedido foi entendido

```bash
python -m src.cli plan "jumps de rock japonês mas nada de streams, 5 a 6 estrelas"
```

Mostra os filtros e preferências sem buscar nada. Ajuste a frase até ficar como você quer.

## 5. Lendo o resultado

- **Caixa "HybridQuery"** no topo: como o pedido foi entendido (filtros, skills desejadas, o que evitar, gêneros). Leia sempre.
- **Linha "catálogo"**: quantos mapsets foram lidos, quantos candidatos sobraram, quantos foram analisados e quantos você já tinha (esses ficam ocultos).
- **`score`**: nota final, de 0 a 1.
- **`sem` / `mec` / `mus`**: aderência ao **sentido** do pedido, à **mecânica** e ao **gênero/clima**.
- **`por quê`**: os motivos, por exemplo `5.29★ 234BPM AR9.2; jumps 0.47; tech 0.80; maior stream 0; gênero: jrock, rock`.
- **Linhas apagadas + aviso amarelo "Filtros relaxados"**: não havia mapas suficientes dentro dos seus números, então os filtros foram afrouxados. Esses itens ficam sempre abaixo dos que cumprem tudo.

## 6. Baixar os mapas e criar a coleção

**Baixar os `.osz`:**

```bash
python -m src.cli recommend "jumps técnicos de rock japonês, 5.2 estrelas" --limit 15 --download out/osz
```

Depois abra cada `.osz` com duplo clique (o osu! importa sozinho) ou mova os arquivos para a pasta `Songs` do osu! e aperte F5 na seleção de músicas.

**Criar uma coleção no jogo** (osu! stable):

1. **Feche o osu!**. Se ele estiver aberto, reescreve o `collection.db` ao sair e apaga a coleção nova.
2. Rode o pedido com `--export` e `--merge` apontando para o `collection.db` do jogo:

```bash
python -m src.cli recommend "..." --limit 25 --name "treino aim" --export "<pasta do osu!>/collection.db" --merge "<pasta do osu!>/collection.db"
```

| Opção | Para que serve |
|---|---|
| `--name` | nome da coleção no jogo |
| `--export` | onde gravar |
| `--merge` | mantém as coleções que você já tem (sem isso, elas somem) |

Antes de sobrescrever, é criado um backup `collection.db.bak-<data>`. A coleção aparece completa assim que os mapas estiverem importados.

Pasta do osu! stable: Windows `%LOCALAPPDATA%\osu!`; Linux (Wine) costuma ser `~/.local/share/osu-wine/osu!`. No **osu!lazer**: Settings → Maintenance → *Import collections from stable*.

## 7. Buscar só nos mapas que você já tem

```bash
python -m src.cli index --songs-dir "<pasta do osu!>/Songs" --no-llm
python -m src.cli recommend --local "mapas técnicos entre 4 e 6 estrelas"
```

Sem `--no-llm`, o Claude descreve cada mapa (gênero e clima melhores, mas cerca de 5 s por mapa). Rodar `index` de novo só processa mapas novos.

## 8. Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `Claude Code não autenticado` | rode `claude`, faça `/login` e depois `/exit`. Até lá, o programa usa as regras fixas. |
| `Catálogo indisponível` | o mirror público não respondeu. Tente de novo em alguns minutos, ou configure a API oficial (próxima linha). |
| quer filtros de gênero/idioma direto na fonte | crie um app OAuth grátis em osu.ppy.sh → Settings → OAuth e coloque `OSU_CLIENT_ID` e `OSU_CLIENT_SECRET` no `.env`. |
| poucos resultados / "Filtros relaxados" | o pedido está restrito demais para o catálogo lido; alargue a faixa de estrelas ou aumente `ONLINE_MAX_SETS` no `.env`. |
| a coleção sumiu no jogo | o osu! estava aberto durante o export. Feche o jogo e exporte de novo, ou restaure um `collection.db.bak-*`. |
| quer ver mapas que já tem | use `--include-owned`. |
