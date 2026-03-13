#!/usr/bin/env python3
# vrptw_ga_complete.py
# Compatibilizado para instâncias Solomon/Homberger (VRPTW).
# Dependências: deap
# Execução: python vrptw_ga_complete.py <instancia.txt>
#
# Modificado: fallback que gera solução aleatória viável caso heurística inicial retorne None.

import sys
import random
import math
import copy
from collections import defaultdict
from deap import base, creator, tools

# ----------------------
# Leitura da instância
# ----------------------
def ler_instancia(caminho_arquivo):
    """
    Lê instância no formato típico Solomon/Homberger:
    Cada linha de cliente: id x y demand ready due service
    Depósito tem id 0.
    Se houver header diferente, adapte a função.
    """
    clientes = []
    deposito = None
    capacidade = None
    num_veiculos = None

    with open(caminho_arquivo, "r") as f:
        linhas = [l.strip() for l in f if l.strip()]

    for l in linhas:
        if "VEHICLE" in l.upper() and "CAPACITY" in l.upper():
            parts = l.replace(":", " ").split()
            nums = [int(p) for p in parts if p.isdigit()]
            if len(nums) >= 2:
                num_veiculos = nums[0]
                capacidade = nums[1]
            break

   
    for l in linhas:
        parts = l.split()
        if len(parts) >= 7:
            try:
                cid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                dem = int(parts[3])
                ready = int(parts[4])
                due = int(parts[5])
                service = int(parts[6])
            except:
                continue
            cliente = {"id": cid, "x": x, "y": y, "demanda": dem, "inicio": ready, "fim": due, "servico": service}
            if cid == 0:
                deposito = cliente
            else:
                clientes.append(cliente)

    if deposito is None:
        # fallback: assume first line is depot
        if clientes:
            deposito = clientes[0]
            clientes = clientes[1:]
        else:
            raise ValueError("Não foi possível encontrar depósito na instância.")

    # if capacity not found, try infer from header lines with 'capacity' word
    if capacidade is None:
        # default grosseiro (ajuste manual recomendado)
        capacidade = max(c["demanda"] for c in clientes) * 5

    if num_veiculos is None:
        # try estimate upper bound: sum(demand)/cap
        total_dem = sum(c["demanda"] for c in clientes)
        num_veiculos = math.ceil(total_dem / capacidade) + 2

    return clientes, deposito, capacidade, num_veiculos

# ----------------------
# Distâncias e mapas
# ----------------------
def distancia_euclid(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return int(round(math.hypot(dx, dy)))

def precompute_matriz(todos):
    n = len(todos)
    mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                mat[i][j] = distancia_euclid(todos[i], todos[j])
    return mat

def construir_mapeamentos(todos):
    id_to_idx = {c["id"]: i for i, c in enumerate(todos)}
    idx_to_id = {i: c["id"] for i, c in enumerate(todos)}
    id_to_cliente = {c["id"]: c for c in todos}
    return id_to_idx, idx_to_id, id_to_cliente

# ----------------------
# Avaliação com janelas (TW-aware)
# ----------------------
def avaliar_rota_tw_by_ids(rota_ids, id_to_cliente):
    """
    rota_ids: lista de ids começando/terminando com depot id.
    Retorna (custo_total, viavel_bool)
    Viabilidade estrita: arrival time <= due. espera permitida.
    """
    custo = 0
    t = 0
    for k in range(len(rota_ids)-1):
        a = id_to_cliente[rota_ids[k]]
        b = id_to_cliente[rota_ids[k+1]]
        d = distancia_euclid(a, b)
        t += d
        # se b for depósito final (id == depot id), não aplicar janela/serviço
        if b["id"] != rota_ids[-1]:
            if t > b["fim"]:
                return (1e9, False)
            if t < b["inicio"]:
                t = b["inicio"]
            t += b["servico"]
        custo += d
    return (custo, True)

# ----------------------
# Dividir cromossomo em rotas viáveis (sequencial, greedy by capacity + TW)
# ----------------------
def dividir_em_rotas_tw_from_sequence(seq_ids, id_to_cliente, capacidade, num_veiculos, depot_id):
    """
    Constrói rotas lendo seq_ids na ordem (heurística greedy).
    Mantém verificação de capacidade e janelas ao inserir no final da rota atual.
    Retorna lista de rotas (listas de ids com depot nas extremidades) ou None se impossível com num_veiculos.
    """
    rotas = []
    rota = [depot_id]
    carga = 0
    tempo_atual = 0  # tempo de chegada para último nó na rota (tempo já incluindo serviço)

    for cid in seq_ids:
        c = id_to_cliente[cid]
        # verifica capacidade
        if carga + c["demanda"] > capacidade:
            # fecha rota atual e inicia nova
            rota.append(depot_id)
            rotas.append(rota)
            rota = [depot_id]
            carga = 0
            tempo_atual = 0

        # simula inserção na rota atual (no final)
        last = id_to_cliente[rota[-1]]
        travel = distancia_euclid(last, c)
        t = tempo_atual + travel
        if t > c["fim"]:
            # cliente não cabe na rota atual (por janela) -> tentar abrir nova rota
            if len(rotas) + 1 >= num_veiculos + 1:  # +1 pois rota atual ainda não foi fechada
                return None
            # fecha rota atual
            rota.append(depot_id)
            rotas.append(rota)
            rota = [depot_id]
            carga = 0
            tempo_atual = 0
            # inserir em nova rota
            travel = distancia_euclid(id_to_cliente[depot_id], c)
            t = travel
            if t > c["fim"]:
                return None  # impossível servir esse cliente a partir do depósito
            if t < c["inicio"]:
                t = c["inicio"]
            t += c["servico"]
            rota.append(cid)
            carga = c["demanda"]
            tempo_atual = t
        else:
            # inserir na rota atual
            if t < c["inicio"]:
                t = c["inicio"]
            t += c["servico"]
            rota.append(cid)
            carga += c["demanda"]
            tempo_atual = t

    rota.append(depot_id)
    rotas.append(rota)

    if len(rotas) > num_veiculos:
        return None
    return rotas

# ----------------------
# Heurística de inserção tipo Solomon I1 (constrói solução inicial viável)
# ----------------------
def heuristica_insercao_solomon(clientes, deposito, capacidade, num_veiculos):
    """
    Implementa heuristic I1: constrói rotas iniciando por semente (menor due?), e insere clientes pelo menor custo de inserção (considerando tempo).
    Retorna rotas (listas de ids com depot nas extremidades) ou None.
    """
    depot_id = deposito["id"]
    remaining = {c["id"]: c for c in clientes}
    rotas = []
    # criterio de semente: earliest due time or largest demand? Vamos usar earliest due among remaining
    while remaining:
        # criar nova rota
        route = [depot_id, depot_id]  # inicialmente só depot start/end
        carga = 0
        tempo_atual = 0
        # escolher semente: cliente com earliest due que caiba
        seed = None
        sorted_remaining = sorted(remaining.values(), key=lambda x: x["fim"])
        for s in sorted_remaining:
            if s["demanda"] <= capacidade:
                # testar se possível servir a partir do depot (tempo)
                travel = distancia_euclid(deposito, s)
                t = travel
                if t <= s["fim"]:
                    seed = s
                    break
        if seed is None:
            # se nenhum cliente cabe, solução inviável
            return None
        # inserir seed (entre depot e depot)
        route = [depot_id, seed["id"], depot_id]
        carga = seed["demanda"]
        # compute tempo_atual after seed
        t = distancia_euclid(deposito, seed)
        if t < seed["inicio"]:
            t = seed["inicio"]
        t += seed["servico"]
        tempo_atual = t
        del remaining[seed["id"]]

        # inserir outros clientes por melhor custo de inserção
        improved = True
        while remaining and improved:
            improved = False
            best_insertion = None
            best_cost_increase = None
            best_new_timeline = None
            # para cada cliente e cada posição entre nodes tentar inserir
            for cid, c in remaining.items():
                if carga + c["demanda"] > capacidade:
                    continue
                for pos in range(1, len(route)):  # posição de inserção entre route[pos-1] -> route[pos]
                    # construir rota temporária
                    temp = route[:pos] + [cid] + route[pos:]
                    # avaliar se rota temp é viável (TW)
                    cost, viable = avaliar_rota_tw_by_ids(temp, {**{depot_id: deposito}, **{cl["id"]: cl for cl in clientes}})
                    if not viable:
                        continue
                    # custo atual da rota
                    cost_curr, _ = avaliar_rota_tw_by_ids(route, {**{depot_id: deposito}, **{cl["id"]: cl for cl in clientes}})
                    delta = cost - cost_curr
                    if best_cost_increase is None or delta < best_cost_increase:
                        best_cost_increase = delta
                        best_insertion = (cid, pos)
            if best_insertion:
                cid, pos = best_insertion
                route = route[:pos] + [cid] + route[pos:]
                carga += remaining[cid]["demanda"]
                del remaining[cid]
                improved = True

        rotas.append(route)
        if len(rotas) > num_veiculos:
            return None

    return rotas

# ----------------------
# 2-opt intra-rota
# ----------------------
def two_opt_route(rota_ids, id_to_cliente):
    best = rota_ids
    best_cost, _ = avaliar_rota_tw_by_ids(best, id_to_cliente)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i+1, len(best) - 1):
                newr = best[:i] + best[i:j][::-1] + best[j:]
                c, viable = avaliar_rota_tw_by_ids(newr, id_to_cliente)
                if viable and c < best_cost:
                    best = newr
                    best_cost = c
                    improved = True
                    break
            if improved:
                break
    return best


def try_best_relocate(rotas, id_to_cliente, capacidade, depot_id):
    best_gain = 1e-9
    best_move = None
    # iterate pairs of routes
    for i, r_from in enumerate(rotas):
        for pos in range(1, len(r_from)-1):
            cid = r_from[pos]
            for j, r_to in enumerate(rotas):
                if i == j:
                    continue
                for insert_pos in range(1, len(r_to)):
                    new_from = r_from[:pos] + r_from[pos+1:]
                    new_to = r_to[:insert_pos] + [cid] + r_to[insert_pos:]
                    # check capacity
                    dem_from = sum(id_to_cliente[x]["demanda"] for x in new_from if x != depot_id)
                    dem_to = sum(id_to_cliente[x]["demanda"] for x in new_to if x != depot_id)
                    if dem_from > capacidade or dem_to > capacidade:
                        continue
                    c_from, v_from = avaliar_rota_tw_by_ids(new_from, id_to_cliente)
                    c_to, v_to = avaliar_rota_tw_by_ids(new_to, id_to_cliente)
                    if not (v_from and v_to):
                        continue
                    old_cost = avaliar_rota_tw_by_ids(r_from, id_to_cliente)[0] + avaliar_rota_tw_by_ids(r_to, id_to_cliente)[0]
                    new_cost = c_from + c_to
                    gain = old_cost - new_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, pos, j, insert_pos, new_from, new_to)
    if best_move:
        i, pos, j, insert_pos, new_from, new_to = best_move
        rotas[i] = new_from
        rotas[j] = new_to
        # remove empty routes (if any, with only depot-depot)
        rotas[:] = [r for r in rotas if not (len(r) == 2 and r[0] == r[1])]
        return True
    return False

# ----------------------
# Exchange (swap two customers between routes)
# ----------------------
def try_best_exchange(rotas, id_to_cliente, capacidade, depot_id):
    best_gain = 1e-9
    best_move = None
    m = len(rotas)
    for i in range(m):
        for j in range(i+1, m):
            r1 = rotas[i]
            r2 = rotas[j]
            for p in range(1, len(r1)-1):
                for q in range(1, len(r2)-1):
                    cid1 = r1[p]
                    cid2 = r2[q]
                    new_r1 = r1[:p] + [cid2] + r1[p+1:]
                    new_r2 = r2[:q] + [cid1] + r2[q+1:]
                    dem1 = sum(id_to_cliente[x]["demanda"] for x in new_r1 if x != depot_id)
                    dem2 = sum(id_to_cliente[x]["demanda"] for x in new_r2 if x != depot_id)
                    if dem1 > capacidade or dem2 > capacidade:
                        continue
                    c1, v1 = avaliar_rota_tw_by_ids(new_r1, id_to_cliente)
                    c2, v2 = avaliar_rota_tw_by_ids(new_r2, id_to_cliente)
                    if not (v1 and v2):
                        continue
                    old_cost = avaliar_rota_tw_by_ids(r1, id_to_cliente)[0] + avaliar_rota_tw_by_ids(r2, id_to_cliente)[0]
                    new_cost = c1 + c2
                    gain = old_cost - new_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, j, p, q, new_r1, new_r2)
    if best_move:
        i, j, p, q, nr1, nr2 = best_move
        rotas[i] = nr1
        rotas[j] = nr2
        return True
    return False

# ----------------------
# Busca local completa sobre rotas (aplica 2-opt, relocate e exchange até convergência)
# ----------------------
def busca_local_completa(rotas, id_to_cliente, capacidade, depot_id, max_passes=5):
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        # 2-opt em cada rota
        for i in range(len(rotas)):
            if len(rotas[i]) > 3:
                nova = two_opt_route(rotas[i], id_to_cliente)
                if nova != rotas[i]:
                    rotas[i] = nova
                    improved = True
        # relocate
        if try_best_relocate(rotas, id_to_cliente, capacidade, depot_id):
            improved = True
        # exchange
        if try_best_exchange(rotas, id_to_cliente, capacidade, depot_id):
            improved = True
        passes += 1
    return rotas

# ----------------------
# LNS: remove k customers e reinserir greedy
# ----------------------
def lns_remove_and_reinsert(rotas, id_to_cliente, capacidade, depot_id, k_remove=5):
    # coletar todos clientes
    clients = [cid for r in rotas for cid in r if cid != depot_id]
    if len(clients) <= k_remove:
        return rotas
    removed = set(random.sample(clients, k_remove))
    # remove them from rotas
    new_rotas = []
    for r in rotas:
        newr = [x for x in r if x not in removed]
        # if route becomes empty (depot-depot) skip or keep
        if len(newr) >= 2:
            new_rotas.append(newr)
    # try reinsert each removed client greedily at best insertion (cost) position where feasible
    for cid in removed:
        best_pos = None
        best_delta = None
        for i, r in enumerate(new_rotas):
            for pos in range(1, len(r)):
                candidate = r[:pos] + [cid] + r[pos:]
                dem = sum(id_to_cliente[x]["demanda"] for x in candidate if x != depot_id)
                if dem > capacidade:
                    continue
                c, v = avaliar_rota_tw_by_ids(candidate, id_to_cliente)
                if not v:
                    continue
                old_c = avaliar_rota_tw_by_ids(r, id_to_cliente)[0]
                delta = c - old_c
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = (i, pos)
        # se não encontrou inserção em rotas existentes, abrir nova rota com o cliente (se possível)
        if best_pos is None:
            # make new route [depot, cid, depot] if feasible
            travel = distancia_euclid(id_to_cliente[depot_id], id_to_cliente[cid])
            if travel <= id_to_cliente[cid]["fim"] and id_to_cliente[cid]["demanda"] <= capacidade:
                new_rotas.append([depot_id, cid, depot_id])
            else:
                # reinserção falhou — tenta colocar em qualquer posição sem quebrar (fallback: append to smallest load route)
                placed = False
                for i, r in enumerate(new_rotas):
                    for pos in range(1, len(r)):
                        candidate = r[:pos] + [cid] + r[pos:]
                        dem = sum(id_to_cliente[x]["demanda"] for x in candidate if x != depot_id)
                        if dem > capacidade:
                            continue
                        c, v = avaliar_rota_tw_by_ids(candidate, id_to_cliente)
                        if v:
                            new_rotas[i] = candidate
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    # last resort: create route even if maybe infeasible (we'll discard later)
                    new_rotas.append([depot_id, cid, depot_id])
        else:
            i, pos = best_pos
            new_rotas[i] = new_rotas[i][:pos] + [cid] + new_rotas[i][pos:]
    # clean up empty routes
    new_rotas = [r for r in new_rotas if not (len(r) == 2 and r[0] == r[1])]
    return new_rotas

# ----------------------
# Funções de avaliação para cromossomo (DEAP)
# ----------------------
def avaliar_individuo_tw(individuo, clientes, deposito, capacidade, num_veiculos, id_to_cliente):
    # individuo contém índices 0..len(clientes)-1 representando permutação de clientes
    seq_ids = [clientes[i]["id"] for i in individuo]
    rotas = dividir_em_rotas_tw_from_sequence(seq_ids, id_to_cliente, capacidade, num_veiculos, deposito["id"])
    if rotas is None:
        return (1e9,)
    total = 0
    for r in rotas:
        c, v = avaliar_rota_tw_by_ids(r, id_to_cliente)
        if not v:
            return (1e9,)
        total += c
    return (total,)

# ----------------------
# DEAP configuration
# ----------------------
def configurar_deap_tw(clientes, deposito, capacidade, num_veiculos, id_to_cliente, seed=42):
    random.seed(seed)
    # safe creator creation
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individuo"):
        creator.create("Individuo", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, list(range(len(clientes))), len(clientes))
    toolbox.register("individuo", tools.initIterate, creator.Individuo, toolbox.indices)
    toolbox.register("populacao", tools.initRepeat, list, toolbox.individuo)

    def avaliar(ind):
        return avaliar_individuo_tw(ind, clientes, deposito, capacidade, num_veiculos, id_to_cliente)

    toolbox.register("evaluate", avaliar)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# ----------------------
# Helpers: converter rotas para sequências cromossomo (list of indices)
# ----------------------
def rotas_para_sequencia_indices(rotas, clientes):
    # rotas: listas de ids (com depot)
    seq_ids = [cid for r in rotas for cid in r if cid != 0]
    indices = [next(i for i, c in enumerate(clientes) if c["id"] == cid) for cid in seq_ids]
    return indices

# ----------------------
# Novas funções: Fallback - gerar solução aleatória viável
# ----------------------
def dividir_por_capacidade(seq_ids, id_to_cliente, capacidade, depot_id):
    """
    Divide ignorando TW: cria rotas baseadas apenas em capacidade.
    Retorna lista de rotas com depot nas extremidades.
    """
    rotas = []
    rota = [depot_id]
    carga = 0
    for cid in seq_ids:
        c = id_to_cliente[cid]
        if carga + c["demanda"] > capacidade:
            rota.append(depot_id)
            rotas.append(rota)
            rota = [depot_id]
            carga = 0
        rota.append(cid)
        carga += c["demanda"]
    rota.append(depot_id)
    rotas.append(rota)
    # remove trivial duplicatas
    rotas = [r for r in rotas if not (len(r) == 2 and r[0] == r[1])]
    return rotas

def gerar_solucao_aleatoria_viavel(clientes, deposito, capacidade, num_veiculos,
                                   id_to_cliente, max_attempts=200, max_vehicle_relax=10):
    """
    Tenta gerar solução viável:
      1) gera permutações aleatórias e testa dividir_em_rotas_tw_from_sequence
      2) se falhar, aumenta temporariamente o número de veículos (até max_vehicle_relax)
      3) como último recurso, divide apenas por capacidade (ignora TW) para garantir alguma solução inicial
    Retorna rotas (não None).
    """
    depot_id = deposito["id"]
    client_ids = [c["id"] for c in clientes]

    # 1) tenta diversas permutações aleatórias mantendo num_veiculos
    for attempt in range(max_attempts):
        random.shuffle(client_ids)
        rotas = dividir_em_rotas_tw_from_sequence(client_ids, id_to_cliente, capacidade, num_veiculos, depot_id)
        if rotas is not None:
            print(f"[fallback] solução aleatória viável encontrada em tentativa {attempt+1} (mesmo num_veiculos).")
            return rotas

    # 2) tenta relaxar o número de veículos (aumenta até max_vehicle_relax)
    for extra in range(1, max_vehicle_relax+1):
        nv = num_veiculos + extra
        for attempt in range(max(50, max_attempts//4)):
            random.shuffle(client_ids)
            rotas = dividir_em_rotas_tw_from_sequence(client_ids, id_to_cliente, capacidade, nv, depot_id)
            if rotas is not None:
                print(f"[fallback] solução viável com num_veiculos aumentado ({nv}) encontrada (extra={extra}).")
                return rotas

    # 3) último recurso: dividir apenas por capacidade (ignora TW) para garantir algo
    print("[fallback] não foi possível gerar solução respeitando TW com tentativas; criando divisão por capacidade (ignora TW).")
    # usar a mesma ordem original (ou embaralhada) — aqui embaralhamos para diversidade
    random.shuffle(client_ids)
    rotas_cap = dividir_por_capacidade(client_ids, id_to_cliente, capacidade, depot_id)
    return rotas_cap

# ----------------------
# Montando tudo no main
# ----------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python vrptw_ga_complete.py <instancia.txt>")
        sys.exit(1)

    arquivo = sys.argv[1]
    clientes, deposito, capacidade, num_veiculos = ler_instancia(arquivo)
    print(f"Clientes: {len(clientes)}, Capacidade: {capacidade}, Veículos (cap est.): {num_veiculos}")

    # construir lista 'todos' com depósito na posição 0
    todos = [deposito] + clientes
    id_to_idx, idx_to_id, id_to_cliente = construir_mapeamentos(todos)

    # heurística inicial (Solomon-like)
    rotas_iniciais = heuristica_insercao_solomon(clientes, deposito, capacidade, num_veiculos)
    if rotas_iniciais is None:
        print("⚠ Heurística inicial falhou — usando fallback (solução aleatória viável).")
        rotas_iniciais = gerar_solucao_aleatoria_viavel(clientes, deposito, capacidade, num_veiculos, id_to_cliente)
        if rotas_iniciais is None:
            # isto não deve acontecer, pois gerar_solucao_aleatoria_viavel devolve algo como última opção
            raise RuntimeError("Falha: não foi possível gerar solução inicial (até o fallback).")

    print("\nSolução inicial heurística (rotas):")
    for i, r in enumerate(rotas_iniciais, 1):
        print(f" Route #{i}: {' '.join(map(str, r[1:-1]))}")

    # configurar DEAP
    toolbox = configurar_deap_tw(clientes, deposito, capacidade, num_veiculos, id_to_cliente, seed=42)

    POP = 200
    NGEN = 500
    CXPB = 0.7
    MUTPB = 0.2
    ELITISM = 2

    # inicializar população
    pop = toolbox.populacao(n=POP)

    # incluir solução heurística como indivíduo
    try:
        seq_indices = rotas_para_sequencia_indices(rotas_iniciais, clientes)
        cw_ind = creator.Individuo(seq_indices)
        cw_ind.fitness.values = toolbox.evaluate(cw_ind)
        pop[0] = cw_ind
    except Exception as e:
        # fallback ignore
        pass

    # avaliar população inicial
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # GA loop com busca local periódica (LNS e relocate)
    best_overall = tools.selBest(pop, 1)[0]
    best_cost = best_overall.fitness.values[0]
    print(f"\nCusto inicial melhor indivíduo: {best_cost:.2f}")

    for gen in range(1, NGEN+1):
        # seleção + clonagem
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        # mutação
        for m in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(m)
                del m.fitness.values
        # avaliar inválidos
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        # elitismo: manter top ELITISM
        elite = tools.selBest(pop, ELITISM)
        pop = offspring
        # substituir piores por elite
        pop_sorted = sorted(pop, key=lambda x: x.fitness.values[0])
        pop_sorted[-ELITISM:] = list(map(toolbox.clone, elite))
        pop = pop_sorted

        # aplicação de busca local periódica
        if gen % 10 == 0 or gen == NGEN:
            # pega melhor indivíduo e aplica LNS + busca local nas rotas correspondentes
            best = tools.selBest(pop, 1)[0]
            seq_ids = [clientes[i]["id"] for i in best]
            rotas = dividir_em_rotas_tw_from_sequence(seq_ids, id_to_cliente, capacidade, num_veiculos, deposito["id"])
            if rotas is not None:
                # aplica lns e local search algumas vezes
                for _ in range(3):
                    rotas = lns_remove_and_reinsert(rotas, id_to_cliente, capacidade, deposito["id"], k_remove=max(3, int(0.05*len(clientes))))
                    rotas = busca_local_completa(rotas, id_to_cliente, capacidade, deposito["id"], max_passes=4)
                # reconverter para indivíduo
                try:
                    new_seq_idx = rotas_para_sequencia_indices(rotas, clientes)
                    new_ind = creator.Individuo(new_seq_idx)
                    new_ind.fitness.values = toolbox.evaluate(new_ind)
                    # substituir se melhor
                    if new_ind.fitness.values[0] < best.fitness.values[0]:
                        # substituir um indivíduo ruim na população
                        worst = tools.selWorst(pop, 1)[0]
                        pop[pop.index(worst)] = new_ind
                except Exception:
                    pass
            else:
                # se a divisão de best falhar por TW, tentamos uma reconstrução aleatória localmente para não travar
                rebuilt = gerar_solucao_aleatoria_viavel(clientes, deposito, capacidade, num_veiculos, id_to_cliente, max_attempts=100, max_vehicle_relax=3)
                try:
                    new_seq_idx = rotas_para_sequencia_indices(rebuilt, clientes)
                    new_ind = creator.Individuo(new_seq_idx)
                    new_ind.fitness.values = toolbox.evaluate(new_ind)
                    worst = tools.selWorst(pop, 1)[0]
                    pop[pop.index(worst)] = new_ind
                except Exception:
                    pass

        # atualizar melhor global
        current_best = tools.selBest(pop, 1)[0]
        if current_best.fitness.values[0] < best_cost:
            best_cost = current_best.fitness.values[0]
            best_overall = current_best
            print(f"Gen {gen}: novo best = {best_cost:.2f}")

    # saída final: reconstruir rotas e aplicar busca final
    seq_ids = [clientes[i]["id"] for i in best_overall]
    rotas = dividir_em_rotas_tw_from_sequence(seq_ids, id_to_cliente, capacidade, num_veiculos, deposito["id"])
    if rotas:
        rotas = busca_local_completa(rotas, id_to_cliente, capacidade, deposito["id"], max_passes=10)
    else:
        # se falhar, tenta fallback final
        rotas = gerar_solucao_aleatoria_viavel(clientes, deposito, capacidade, num_veiculos, id_to_cliente)

    print("\nMelhor solução encontrada:")
    print(f"Custo total: {best_overall.fitness.values[0]:.2f}\n")
    for i, r in enumerate(rotas, 1):
        print(f"Route #{i}: {' '.join(map(str, r[1:-1]))}")

if __name__ == "__main__":
    main()
