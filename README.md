# VRPTW — Algoritmo Genético Híbrido

> Projeto de Iniciação Científica (PIBIC/FAPEMIG) — Universidade Federal de Viçosa  
> Orientador: Prof. Marcus Henrique Soares Mendes

Implementação de um **Algoritmo Genético híbrido com busca local** para o **Problema de Roteamento de Veículos com Janelas de Tempo (VRPTW)**, testado sobre as instâncias clássicas de benchmark de Solomon e Homberger.


## Sumário

1. [Sobre o problema](#sobre-o-problema)
2. [Metodologia](#metodologia)
3. [Estrutura do repositório](#estrutura-do-repositório)
4. [Instalação](#instalação)
5. [Uso — Algoritmo principal](#uso--algoritmo-principal-vrptw_ga_v2py)
6. [Uso — Validador comparativo](#uso--validador-comparativo-validarpy)
7. [Formato das instâncias](#formato-das-instâncias)
8. [Formato dos arquivos de solução ótima (.sol)](#formato-dos-arquivos-de-solução-ótima-sol)
9. [Parâmetros e ajuste fino](#parâmetros-e-ajuste-fino)
10. [Referências](#referências)


## Sobre o problema

O **VRPTW** (Vehicle Routing Problem with Time Windows) é um problema de otimização combinatória NP-difícil que consiste em determinar um conjunto de rotas de custo mínimo para uma frota de veículos que atende um conjunto de clientes, respeitando:

- **Capacidade** máxima de cada veículo
- **Janelas de tempo** `[a_i, b_i]` para o início do atendimento de cada cliente *i*
- **Objetivo hierárquico** (padrão Solomon):
  1. Minimizar a **distância total** percorrida



## Metodologia

O método implementado combina três componentes principais:

### 1. Construção da solução inicial — Heurística I1 de Solomon

A população inicial é gerada pela heurística de inserção sequencial **I1**, proposta por Solomon (1987). O processo consiste em:

1. Selecionar um cliente-semente para iniciar cada rota (critérios: `due date`, `distância do depósito`, ou `aleatório`)
2. Inserir iterativamente o cliente de menor custo de inserção viável na rota corrente
3. Abrir uma nova rota quando nenhum cliente restante puder ser inserido

Para diversificar a população, as soluções subsequentes são geradas por **perturbação** das soluções base: remove-se um subconjunto de clientes aleatoriamente e eles são reinseridos de forma gulosa.

### 2. Algoritmo Genético (DEAP)

O GA opera sobre uma representação de **lista de rotas** e utiliza:

| Componente | Descrição |
|---|---|
| **Crossover** | Route-Based Crossover (RBX): copia a melhor rota de um pai e reinsere os clientes restantes do outro |
| **Mutação 1** | Or-opt: realoca segmentos de 1–2 clientes para melhor posição |
| **Mutação 2** | Reinserção aleatória: remove um cliente e o reinsere na melhor posição viável |
| **Mutação 3** | Embaralhamento de rota: permuta a ordem interna de uma rota e corrige com Or-opt |
| **Seleção** | Torneio de tamanho 3 |
| **Elitismo** | Os *k* melhores indivíduos são sempre preservados para a próxima geração |
| **Fitness** | `nv × 10⁶ + distância_total` — garante hierarquia: reduzir veículos tem prioridade máxima |

### 3. Busca Local

A busca local é aplicada periodicamente ao melhor indivíduo da população e ao final da execução, com o seguinte pipeline:

1. **Eliminação de rotas** (`tentar_eliminar_rota`): tenta realocar todos os clientes da menor rota nas demais, eliminando um veículo
2. **Or-opt inter-rotas** (segmentos de tamanho 1, 2 e 3): move segmentos entre rotas diferentes
3. **2-opt intra-rota**: inverte subsequências dentro de cada rota
4. **Node Relocate**: move clientes individuais para a melhor posição em outra rota
5. Segunda rodada de Or-opt após relocate



## Estrutura do repositório

```
.
├── Data/
│   ├── Solomon/                # Instâncias Solomon (100 clientes)
│   │   ├── C101.txt            # Dados da instância (coordenadas, demandas, janelas de tempo)
│   │   ├── C101.sol            # Solução ótima conhecida da literatura
│   │   └── ...
│   └── Homberger/              # Instâncias Homberger (200–1000 clientes)
│       ├── C1_2_1.txt
│       └── ...
│
├── src/
│   ├── vrp/
│   │   └── vrp.py      # Algoritmo principal (GA + Busca Local)
│   └── validar/
│       └── validar.py          # Executor em lote + tabela comparativa
│
├── Homberger.html              # Soluções ótimas da literatura para Homberger
└── README.md
```



## Instalação

### Requisitos

- Python 3.8+
- pip

### Dependências

```bash
pip install deap rich
```

| Pacote | Uso |
|---|---|
| `deap` | Framework do Algoritmo Genético (obrigatório) |
| `rich` | Tabela colorida no terminal pelo `validar.py` (opcional) |


## Uso — Algoritmo principal (`vrptw_ga_v2.py`)

Executa o GA em uma única instância.

```bash
python vrptw_ga_v2.py <instancia.txt> [opções]
```

### Exemplos

```bash
# Execução básica com parâmetros padrão (120s, pop=200, gen=500)
python vrptw_ga_v2.py Data/Solomon/C101.txt

# Execução com tempo estendido e semente fixa
python vrptw_ga_v2.py Data/Solomon/R101.txt --time 300 --pop 300 --seed 7

# Execução rápida para teste
python vrptw_ga_v2.py Data/Solomon/C101.txt --time 30 --pop 50 --gen 100
```

### Parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `instancia` | — | Caminho para o arquivo `.txt` da instância (obrigatório) |
| `--time` | `120` | Limite de tempo total em segundos |
| `--pop` | `200` | Tamanho da população |
| `--gen` | `500` | Número máximo de gerações |
| `--seed` | `42` | Semente aleatória (para reprodutibilidade) |
| `--ls_interval` | `10` | Aplicar busca local a cada N gerações |

### Saída esperada

```
Instância  : Data/Solomon/C101.txt
Clientes   : 100
Capacidade : 200
Veículos   : 25
Tempo max  : 120s | Pop: 200 | Gen: 500

Gerando população inicial (Solomon I1)...
Melhor inicial: 12 veículos | dist=1234.56 | fit=12001234.56
  Gen   10 | veículos=11 | dist=1051.20 | t=15.3s
  Gen   40 | veículos=10 | dist=828.94  | t=52.1s

Busca local final...

==================================================
MELHOR SOLUÇÃO ENCONTRADA
  Veículos   : 10
  Distância  : 828.94
==================================================
Route #1: 5 3 7 8 10 11 9 6 4 2 1
...
```



## Uso — Validador comparativo (`validar.py`)

Executa o algoritmo em **todas as instâncias** de uma pasta, compara com as soluções ótimas dos arquivos `.sol` e gera uma tabela comparativa.

O `validar.py` deve estar no mesmo diretório que `vrp.py`, ou o caminho do script deve ser informado via `--script`.

```bash
python validar.py [opções]
```

### Exemplos

```bash
# Execução padrão: roda todas as instâncias em Data/Solomon com 120s cada
python validar.py

# Apenas lê os .sol e metadados — NÃO executa o GA (útil para inspecionar a pasta)
python validar.py --only-parse

# Instâncias em pasta diferente
python validar.py --dir Data/Homberger

# GA mais rápido por instância (útil para testes rápidos)
python validar.py --time 60 --pop 100 --gen 200

# Sem rich (tabela ASCII pura, sem dependências externas)
python validar.py --no-rich
```

### Parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--dir` | `Data/Solomon` | Pasta contendo os pares `.sol`/`.txt` |
| `--script` | `vrptw_ga_v2.py` | Caminho para o script do GA |
| `--time` | `120` | Limite de tempo por instância (segundos) |
| `--pop` | `200` | Tamanho da população do GA |
| `--gen` | `500` | Número máximo de gerações |
| `--seed` | `42` | Semente aleatória |
| `--only-parse` | `False` | Apenas lê os `.sol`, sem rodar o GA |
| `--csv` | `resultados_solomon.csv` | Arquivo CSV de saída |

### Saída — Tabela comparativa

O validador gera automaticamente uma tabela com as seguintes colunas:

| Coluna | Descrição |
|---|---|
| **Instância** | Nome da instância (ex: `C101`) |
| **Clientes** | Número de clientes da instância |
| **Capacidade** | Capacidade máxima de cada veículo |
| **Veíc. Disp.** | Número de veículos disponíveis (do arquivo `.txt`) |
| **Veíc. GA** | Número de veículos utilizados pela solução do GA |
| **Veíc. Ótimo** | Número de veículos da solução ótima (do `.sol`) |
| **Custo GA** | Distância total percorrida pelo GA |
| **Custo Ótimo** | Distância total da solução ótima |
| **Desvio %** | `(Custo GA − Custo Ótimo) / Custo Ótimo × 100` |

A coluna **Desvio %** é colorida quando `rich` está instalado:

- 🟢 **Verde** — desvio ≤ 0% (igual ou melhor que o ótimo)
- 🟡 **Amarelo** — desvio entre 0% e 5%
- 🔴 **Vermelho** — desvio > 5%

Ao final da tabela é exibido o **desvio médio** sobre todas as instâncias executadas.

### Arquivo CSV de saída

Além da tabela no terminal, o validador salva automaticamente um arquivo `resultados_solomon.csv` com todas as colunas, podendo ser aberto no Excel ou LibreOffice para análise posterior.

### Estrutura esperada da pasta de instâncias

O validador emparelha arquivos pelo nome base (stem). Para cada instância, devem existir **dois arquivos com o mesmo nome**:

```
Data/Solomon/
├── C101.sol       ← solução ótima da literatura
├── C101.txt       ← dados da instância
├── C102.sol
├── C102.txt
├── R101.sol
├── R101.txt
└── ...
```

Arquivos `.sol` sem `.txt` correspondente (ou vice-versa) são ignorados com aviso.


## Formato das instâncias

Formato padrão Solomon/Homberger (texto simples):

```
VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
    0       40       50        0        0         1236         0       ← depósito (id=0)
    1       45       68       10       912          967        90
    2       45       70       30      825          870        90
   ...
```

- A linha com dois inteiros `K C` define o número de veículos disponíveis e a capacidade
- O cliente `0` é sempre o depósito
- Coordenadas e tempos são inteiros ou reais



## Formato dos arquivos de solução ótima (`.sol`)

```
Route #1: 5 3 7 8 10 11 9 6 4 2 1 75
Route #2: 13 17 18 19 15 16 14 12
...
Route #10: 98 96 95 94 92 93 97 100 99
Cost 827.3
```

- Cada linha `Route #N:` lista os IDs dos clientes visitados (sem o depósito)
- A linha `Cost` contém a distância total da solução ótima
- O número de rotas define o número mínimo de veículos necessários


## Parâmetros e ajuste fino

### Instâncias Solomon (100 clientes)

| Cenário | `--time` | `--pop` | `--gen` |
|---|---|---|---|
| Teste rápido | 30s | 50 | 100 |
| Execução padrão | 120s | 200 | 500 |
| Alta qualidade | 300s | 300 | 1000 |

### Instâncias Homberger (200–1000 clientes)

| Tamanho | `--time` sugerido | `--pop` sugerido |
|---|---|---|
| 200 clientes | 300s | 150 |
| 400 clientes | 600s | 100 |
| 600–1000 clientes | 1200s+ | 80 |

> **Nota:** O tempo de construção da população inicial cresce com O(n²) por instância. Para instâncias grandes, reduza `--pop` e aumente `--time`.



## Referências

- **Solomon, M. M.** (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254–265.
- **Homberger, J., & Gehring, H.** (2005). A two-phase hybrid metaheuristic for the vehicle routing problem with time windows. *European Journal of Operational Research*, 162(1), 220–238.
- **Fortin, F.-A. et al.** (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research*, 13, 2171–2175.
- Benchmark instances: [Solomon's VRPTW instances](http://web.cba.neu.edu/~msolomon/problems.htm)
- Best known solutions: [SINTEF repository](https://www.sintef.no/projectweb/top/vrptw/)
