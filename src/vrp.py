#!/usr/bin/env python3
# vrptw_ga_v2.py  —  Genetic Algorithm + Local Search para VRPTW
# Compatível com instâncias Solomon (100) e Homberger (200-1000 clientes).
#
# Objetivo hierárquico (padrão da literatura Solomon):
#   1° minimizar número de veículos
#   2° minimizar distância total
#
# Dependências: deap
#   pip install deap
#
# Execução:
#   python vrptw_ga_v2.py <instancia.txt>
#   python vrptw_ga_v2.py <instancia.txt> --time 120 --pop 300 --seed 42

import sys
import math
import random
import copy
import time
import argparse
from typing import List, Dict, Optional, Tuple

from deap import base, creator, tools

# ══════════════════════════════════════════════════════════════════════════════
# LEITURA DE INSTÂNCIA
# ══════════════════════════════════════════════════════════════════════════════

def ler_instancia(path: str):
    clientes, deposito, capacidade, num_veiculos = [], None, None, None
    with open(path) as f:
        linhas = [l.strip() for l in f if l.strip()]

    for l in linhas:
        up = l.upper()
        if "NUMBER" in up and "CAPACITY" in up:
            continue
        parts = l.split()
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                if 1 <= a <= 1000 and 1 <= b <= 100_000:
                    num_veiculos, capacidade = a, b
                    break
            except ValueError:
                pass

    for l in linhas:
        parts = l.split()
        if len(parts) < 7:
            continue
        try:
            cid = int(parts[0])
            x   = float(parts[1]);  y   = float(parts[2])
            dem = int(parts[3])
            rdy = int(parts[4]);    due = int(parts[5])
            srv = int(parts[6])
        except ValueError:
            continue
        c = dict(id=cid, x=x, y=y, demanda=dem, inicio=rdy, fim=due, servico=srv)
        if cid == 0:
            deposito = c
        else:
            clientes.append(c)

    if deposito is None and clientes:
        deposito, clientes = clientes[0], clientes[1:]
    if capacidade is None:
        capacidade = max(c["demanda"] for c in clientes) * 5
    if num_veiculos is None:
        num_veiculos = math.ceil(sum(c["demanda"] for c in clientes) / capacidade) + 5

    return clientes, deposito, capacidade, num_veiculos


# ══════════════════════════════════════════════════════════════════════════════
# DISTÂNCIA (cache por par de ids)
# ══════════════════════════════════════════════════════════════════════════════

_dcache: Dict[Tuple[int,int], float] = {}

def dist(a: dict, b: dict) -> float:
    key = (a["id"], b["id"])
    v = _dcache.get(key)
    if v is None:
        v = math.hypot(a["x"]-b["x"], a["y"]-b["y"])
        _dcache[key] = v
        _dcache[(b["id"], a["id"])] = v
    return v


# ══════════════════════════════════════════════════════════════════════════════
# REPRESENTAÇÃO: cromossomo = lista de rotas
# Cada rota é uma lista de IDs de clientes (sem depósito nas extremidades).
# O cromossomo em si é um objeto Python list-of-lists encapsulado em
# creator.Individuo. O DEAP opera sobre ele via operadores customizados.
# ══════════════════════════════════════════════════════════════════════════════

# ── Verificação TW e capacidade ────────────────────────────────────────────

def route_feasible(route: List[int], cmap: Dict[int,dict], depot: dict, cap: int) -> bool:
    load = 0
    t = 0.0
    prev = depot
    for cid in route:
        c = cmap[cid]
        load += c["demanda"]
        if load > cap:
            return False
        t += dist(prev, c)
        if t > c["fim"]:
            return False
        if t < c["inicio"]:
            t = c["inicio"]
        t += c["servico"]
        prev = c
    return True

def route_dist(route: List[int], cmap: Dict[int,dict], depot: dict) -> float:
    if not route:
        return 0.0
    d = dist(depot, cmap[route[0]])
    for i in range(len(route)-1):
        d += dist(cmap[route[i]], cmap[route[i+1]])
    d += dist(cmap[route[-1]], depot)
    return d

def solution_cost(routes: List[List[int]], cmap: Dict[int,dict], depot: dict) -> Tuple[int,float]:
    """Retorna (num_veiculos, dist_total). Objetivo hierárquico."""
    td = sum(route_dist(r, cmap, depot) for r in routes)
    return (len(routes), td)

def solution_feasible(routes: List[List[int]], cmap: Dict[int,dict], depot: dict, cap: int) -> bool:
    return all(route_feasible(r, cmap, depot, cap) for r in routes)

# ── Custo de inserção de um cliente em posição pos ─────────────────────────

def insertion_delta(route: List[int], cid: int, pos: int,
                    cmap: Dict[int,dict], depot: dict) -> float:
    c = cmap[cid]
    prev = depot if pos == 0 else cmap[route[pos-1]]
    nxt  = depot if pos == len(route) else cmap[route[pos]]
    return dist(prev, c) + dist(c, nxt) - dist(prev, nxt)

def best_insertion(route: List[int], cid: int,
                   cmap: Dict[int,dict], depot: dict, cap: int) -> Optional[Tuple[float,int]]:
    """
    Melhor posição de inserção viável. Retorna (delta, pos) ou None.
    Verificação de TW incremental: O(n) por posição em vez de recriar rota.
    """
    c = cmap[cid]

    # Verifica capacidade primeiro (barato)
    load = sum(cmap[x]["demanda"] for x in route) + c["demanda"]
    if load > cap:
        return None

    # Pré-computa tempos de chegada para cada posição da rota original
    # times[i] = tempo de chegada no cliente route[i] (já com espera, antes do serviço)
    # departures[i] = tempo de saída de route[i] (após serviço)
    n = len(route)
    arrivals  = [0.0] * n   # chegada em route[i]
    departures= [0.0] * n   # saída de route[i]

    t = 0.0
    prev = depot
    for i, rid in enumerate(route):
        rc = cmap[rid]
        t += dist(prev, rc)
        arrivals[i] = t
        if t < rc["inicio"]:
            t = rc["inicio"]
        t += rc["servico"]
        departures[i] = t
        prev = rc

    best_d, best_p = None, None

    for pos in range(n + 1):
        # Verifica TW do novo cliente na posição `pos`
        prev_node = depot if pos == 0 else cmap[route[pos-1]]
        t_arrive_new = (0.0 if pos == 0 else departures[pos-1]) + dist(prev_node, c)

        if t_arrive_new > c["fim"]:
            continue  # chegou tarde demais

        t_wait = max(t_arrive_new, c["inicio"])
        t_depart_new = t_wait + c["servico"]

        # Verifica propagação de atraso nos clientes seguintes
        feasible = True
        t_prop = t_depart_new
        for j in range(pos, n):
            rj = cmap[route[j]]
            prev_j = c if j == pos else cmap[route[j-1]]
            t_arr_j = t_prop + dist(prev_j, rj)
            if t_arr_j > rj["fim"]:
                feasible = False
                break
            t_wait_j = max(t_arr_j, rj["inicio"])
            t_prop = t_wait_j + rj["servico"]
            # Se não há atraso extra (chegamos antes do que sem inserção), para cedo
            if j < n and t_prop <= departures[j]:
                break   # restante da rota não é afetado

        if not feasible:
            continue

        d = insertion_delta(route, cid, pos, cmap, depot)
        if best_d is None or d < best_d:
            best_d, best_p = d, pos

    return None if best_p is None else (best_d, best_p)


# ══════════════════════════════════════════════════════════════════════════════
# HEURÍSTICA INICIAL: Solomon I1 (múltiplas sementes)
# ══════════════════════════════════════════════════════════════════════════════

def solomon_i1(clientes: List[dict], deposito: dict,
               capacidade: int, cmap: Dict[int,dict],
               seed_criterion: str = "due") -> List[List[int]]:
    """
    Constrói solução via I1. seed_criterion: 'due' | 'far' | 'random'
    """
    remaining = {c["id"]: c for c in clientes}
    routes: List[List[int]] = []

    while remaining:
        # escolhe semente
        if seed_criterion == "due":
            candidates = sorted(remaining.values(), key=lambda x: x["fim"])
        elif seed_criterion == "far":
            candidates = sorted(remaining.values(),
                                key=lambda x: -dist(deposito, x))
        else:
            candidates = list(remaining.values())
            random.shuffle(candidates)

        seed = None
        for s in candidates:
            if route_feasible([s["id"]], cmap, deposito, capacidade):
                seed = s
                break
        if seed is None:
            seed = list(remaining.values())[0]

        route = [seed["id"]]
        del remaining[seed["id"]]

        improved = True
        while improved and remaining:
            improved = False
            best_cid, best_pos, best_d = None, None, float("inf")
            for cid in remaining:
                res = best_insertion(route, cid, cmap, deposito, capacidade)
                if res and res[0] < best_d:
                    best_d, best_pos, best_cid = res[0], res[1], cid
            if best_cid is not None:
                route.insert(best_pos, best_cid)
                del remaining[best_cid]
                improved = True

        routes.append(route)

    return routes


def perturbar_solucao(routes: List[List[int]], cmap, depot, cap,
                      n_remove: int = 5) -> List[List[int]]:
    """
    Perturbação rápida: remove n_remove clientes aleatórios e reinseere greedy.
    Muito mais rápido que rodar I1 do zero para gerar diversidade na população.
    """
    routes = [list(r) for r in routes if r]
    all_ids = [cid for r in routes for cid in r]
    if len(all_ids) <= n_remove:
        return routes
    removed = random.sample(all_ids, n_remove)
    removed_set = set(removed)
    new_routes = []
    for r in routes:
        nr = [x for x in r if x not in removed_set]
        if nr:
            new_routes.append(nr)
    random.shuffle(removed)
    for cid in removed:
        best_d, best_ri, best_pos = None, None, None
        for ri, r in enumerate(new_routes):
            res = best_insertion(r, cid, cmap, depot, cap)
            if res is None:
                continue
            if best_d is None or res[0] < best_d:
                best_d, best_ri, best_pos = res[0], ri, res[1]
        if best_ri is not None:
            new_routes[best_ri].insert(best_pos, cid)
        else:
            new_routes.append([cid])
    return [r for r in new_routes if r]


def gerar_populacao_inicial(clientes, deposito, capacidade, cmap,
                            pop_size: int, seed_val: int) -> List[List[List[int]]]:
    """
    Gera pop_size indivíduos:
    - 3 soluções base via Solomon I1 (due, far, random)  — O(n²) cada
    - Restante via perturbação das bases — muito mais rápido
    Resolve o problema de 'Limite de tempo atingido na geração 1'.
    """
    pop = []
    random.seed(seed_val)
    n_remove = max(3, len(clientes) // 10)

    bases = []
    for criterion in ["due", "far", "random"]:
        routes = solomon_i1(clientes, deposito, capacidade, cmap, criterion)
        bases.append([list(r) for r in routes])
        pop.append([list(r) for r in routes])

    while len(pop) < pop_size:
        base = random.choice(bases)
        perturbed = perturbar_solucao(base, cmap, deposito, capacidade, n_remove)
        pop.append(perturbed)

    return pop[:pop_size]


# ══════════════════════════════════════════════════════════════════════════════
# BUSCA LOCAL
# ══════════════════════════════════════════════════════════════════════════════

def two_opt_intra(routes: List[List[int]], cmap, depot, cap) -> List[List[int]]:
    """2-opt dentro de cada rota."""
    routes = [list(r) for r in routes]
    for ri in range(len(routes)):
        r = routes[ri]
        improved = True
        while improved:
            improved = False
            for i in range(len(r)-1):
                for j in range(i+1, len(r)):
                    new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
                    if (route_feasible(new_r, cmap, depot, cap) and
                            route_dist(new_r, cmap, depot) < route_dist(r, cmap, depot) - 1e-9):
                        r = new_r
                        improved = True
                        break
                if improved:
                    break
            routes[ri] = r
    return routes


def or_opt(routes: List[List[int]], cmap, depot, cap,
           seg_sizes=(1, 2, 3), inter_route=True) -> List[List[int]]:
    """
    Or-opt: move segmentos de tamanho k para melhor posição
    (intra e inter rotas). Mais eficaz que 2-opt para VRPTW.
    """
    routes = [list(r) for r in routes]
    improved_global = True

    while improved_global:
        improved_global = False
        for seg_size in seg_sizes:
            for ri in range(len(routes)):
                r = routes[ri]
                if len(r) <= seg_size:
                    continue
                i = 0
                while i <= len(r) - seg_size:
                    seg = r[i:i+seg_size]
                    without = r[:i] + r[i+seg_size:]

                    best_gain = 1e-9
                    best_move = None
                    old_cost_ri = route_dist(r, cmap, depot)

                    # tentativas intra-rota
                    target_list = [(ri, without)]
                    # tentativas inter-rotas
                    if inter_route:
                        for rj in range(len(routes)):
                            if rj != ri:
                                target_list.append((rj, routes[rj]))

                    for rj, r_target in target_list:
                        for j in range(len(r_target)+1):
                            new_target = r_target[:j] + seg + r_target[j:]
                            if not route_feasible(new_target, cmap, depot, cap):
                                continue
                            if rj == ri:
                                gain = old_cost_ri - route_dist(new_target, cmap, depot)
                            else:
                                old_cost_rj = route_dist(r_target, cmap, depot)
                                new_from_cost = route_dist(without, cmap, depot) if without else 0.0
                                gain = (old_cost_ri + old_cost_rj
                                        - new_from_cost
                                        - route_dist(new_target, cmap, depot))
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (rj, j, new_target,
                                             without if rj != ri else None)

                    if best_move:
                        rj, j, new_target, new_from = best_move
                        if rj == ri:
                            routes[ri] = new_target
                            r = new_target
                        else:
                            routes[ri] = new_from
                            routes[rj] = new_target
                            r = routes[ri]
                        improved_global = True
                        i = 0  # reinicia para rota modificada
                        continue
                    i += 1

    return [r for r in routes if r]


def node_relocate(routes: List[List[int]], cmap, depot, cap) -> List[List[int]]:
    """Move um cliente para melhor posição em outra rota."""
    routes = [list(r) for r in routes]
    improved = True
    while improved:
        improved = False
        for ri in range(len(routes)):
            for pos in range(len(routes[ri])):
                cid = routes[ri][pos]
                without = routes[ri][:pos] + routes[ri][pos+1:]
                best_gain = 1e-9
                best_move = None
                old_cost_ri = route_dist(routes[ri], cmap, depot)

                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    res = best_insertion(routes[rj], cid, cmap, depot, cap)
                    if res is None:
                        continue
                    d, ins_pos = res
                    old_cost_rj = route_dist(routes[rj], cmap, depot)
                    new_from_cost = route_dist(without, cmap, depot) if without else 0.0
                    new_to = routes[rj][:ins_pos] + [cid] + routes[rj][ins_pos:]
                    new_to_cost = route_dist(new_to, cmap, depot)
                    gain = old_cost_ri + old_cost_rj - new_from_cost - new_to_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (rj, ins_pos, without, new_to)

                if best_move:
                    rj, ins_pos, new_from, new_to = best_move
                    routes[ri] = new_from
                    routes[rj] = new_to
                    routes = [r for r in routes if r]
                    improved = True
                    break
            if improved:
                break
    return routes


def tentar_eliminar_rota(routes: List[List[int]], cmap, depot, cap) -> List[List[int]]:
    """
    Tenta realocar todos os clientes da menor rota nas demais.
    Se conseguir, elimina a rota (reduz número de veículos).
    Repetido até não haver mais eliminações.
    """
    routes = [list(r) for r in routes]
    changed = True
    while changed:
        changed = False
        routes.sort(key=len)  # tenta eliminar as menores primeiro
        for ri in range(len(routes)):
            to_remove = list(routes[ri])
            remaining_routes = [list(r) for r in routes if r is not routes[ri]]
            temp_routes = [list(r) for r in remaining_routes]
            success = True
            for cid in to_remove:
                placed = False
                best_d, best_rj, best_pos = None, None, None
                for rj, r in enumerate(temp_routes):
                    res = best_insertion(r, cid, cmap, depot, cap)
                    if res is None:
                        continue
                    if best_d is None or res[0] < best_d:
                        best_d, best_rj, best_pos = res[0], rj, res[1]
                if best_rj is not None:
                    temp_routes[best_rj].insert(best_pos, cid)
                    placed = True
                if not placed:
                    success = False
                    break
            if success:
                routes = temp_routes
                changed = True
                break
    return [r for r in routes if r]


def busca_local_completa(routes: List[List[int]], cmap, depot, cap,
                         time_limit_s: float = 10.0,
                         eliminar_rotas: bool = True) -> List[List[int]]:
    """Pipeline completo de busca local."""
    t0 = time.time()
    routes = [list(r) for r in routes if r]

    # 1. Tentar eliminar rotas (prioridade máxima: reduz veículos)
    if eliminar_rotas:
        routes = tentar_eliminar_rota(routes, cmap, depot, cap)

    if time.time() - t0 >= time_limit_s:
        return routes

    # 2. Or-opt inter-rotas (mais poderoso)
    routes = or_opt(routes, cmap, depot, cap, seg_sizes=(1, 2, 3), inter_route=True)

    if time.time() - t0 >= time_limit_s:
        return routes

    # 3. 2-opt intra
    routes = two_opt_intra(routes, cmap, depot, cap)

    if time.time() - t0 >= time_limit_s:
        return routes

    # 4. Relocate inter-rotas
    routes = node_relocate(routes, cmap, depot, cap)

    # 5. Segunda rodada de or-opt após relocate
    if time.time() - t0 < time_limit_s:
        routes = or_opt(routes, cmap, depot, cap, seg_sizes=(1, 2), inter_route=True)

    return [r for r in routes if r]


# ══════════════════════════════════════════════════════════════════════════════
# OPERADORES GENÉTICOS (DEAP)
# ══════════════════════════════════════════════════════════════════════════════

def rbx_crossover(ind1, ind2, cmap, depot, cap):
    """
    Route-Based Crossover (RBX) para VRPTW.
    1. Escolhe uma rota de ind1 que caiba em ind2.
    2. Copia a rota para o filho.
    3. Remove os clientes da rota copiada das demais rotas do pai2.
    4. Reinserive clientes faltantes pelo método greedy.
    Preserva a estrutura de rotas, o que o OX não faz.
    """
    def apply_rbx(parent_donor, parent_receiver):
        if not parent_donor:
            return [list(r) for r in parent_receiver]

        # Escolhe rota do doador com maior potencial (menor custo por cliente)
        scored = sorted(parent_donor,
                        key=lambda r: route_dist(r, cmap, depot) / max(len(r), 1))
        chosen_route = list(scored[0])
        chosen_set = set(chosen_route)

        # Começa com a rota escolhida + rotas do receptor sem clientes duplicados
        child_routes = [list(chosen_route)]
        for r in parent_receiver:
            nr = [x for x in r if x not in chosen_set]
            if nr:
                child_routes.append(nr)

        # Verifica clientes faltantes e reinsere
        all_clients = {cid for r in parent_receiver for cid in r}
        present = {cid for r in child_routes for cid in r}
        missing = list(all_clients - present)
        random.shuffle(missing)

        for cid in missing:
            best_d, best_ri, best_pos = None, None, None
            for ri, r in enumerate(child_routes):
                res = best_insertion(r, cid, cmap, depot, cap)
                if res is None:
                    continue
                if best_d is None or res[0] < best_d:
                    best_d, best_ri, best_pos = res[0], ri, res[1]
            if best_ri is not None:
                child_routes[best_ri].insert(best_pos, cid)
            else:
                child_routes.append([cid])

        return [r for r in child_routes if r]

    child1_routes = apply_rbx(ind1, ind2)
    child2_routes = apply_rbx(ind2, ind1)

    # Atualiza in-place (DEAP opera sobre listas)
    ind1[:] = child1_routes
    ind2[:] = child2_routes
    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2


def mutation_or_opt(ind, cmap, depot, cap, prob_seg=0.5):
    """
    Mutação: aplica um passo de Or-opt (move 1 ou 2 clientes).
    Mais informada que shuffle aleatório.
    """
    if len(ind) < 2:
        return ind,
    routes = [list(r) for r in ind]
    # Escolhe tamanho de segmento aleatoriamente
    seg = 1 if random.random() < prob_seg else 2
    routes = or_opt(routes, cmap, depot, cap, seg_sizes=(seg,), inter_route=True)
    ind[:] = [r for r in routes if r]
    del ind.fitness.values
    return ind,


def mutation_route_shuffle(ind, cmap, depot, cap):
    """
    Mutação: embaralha a ordem interna de uma rota aleatória
    e corrige com Or-opt local.
    """
    if not ind:
        return ind,
    routes = [list(r) for r in ind]
    ri = random.randint(0, len(routes)-1)
    random.shuffle(routes[ri])
    # corrige TW com or-opt intra
    routes = or_opt(routes, cmap, depot, cap, seg_sizes=(1,), inter_route=False)
    ind[:] = [r for r in routes if r]
    del ind.fitness.values
    return ind,


def mutation_insert_random(ind, cmap, depot, cap):
    """
    Mutação: remove um cliente aleatório e reinsere na melhor posição viável.
    """
    routes = [list(r) for r in ind]
    all_ids = [cid for r in routes for cid in r]
    if not all_ids:
        return ind,
    cid = random.choice(all_ids)
    # remove
    for r in routes:
        if cid in r:
            r.remove(cid)
    routes = [r for r in routes if r]
    # reinsere na melhor posição
    best_d, best_ri, best_pos = None, None, None
    for ri, r in enumerate(routes):
        res = best_insertion(r, cid, cmap, depot, cap)
        if res is None:
            continue
        if best_d is None or res[0] < best_d:
            best_d, best_ri, best_pos = res[0], ri, res[1]
    if best_ri is not None:
        routes[best_ri].insert(best_pos, cid)
    else:
        routes.append([cid])
    ind[:] = [r for r in routes if r]
    del ind.fitness.values
    return ind,


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE AVALIAÇÃO (DEAP)
# ══════════════════════════════════════════════════════════════════════════════

# Penalidade para soluções inviáveis
PENALTY_INFEASIBLE = 1e9

def avaliar(ind, cmap, depot, cap):
    """
    Fitness hierárquico: (num_veiculos * BIG + dist_total,)
    Assim o DEAP minimiza veículos antes de distância.
    Soluções inviáveis recebem penalidade enorme.
    """
    routes = list(ind)
    if not routes:
        return (PENALTY_INFEASIBLE,)
    for r in routes:
        if not route_feasible(r, cmap, depot, cap):
            return (PENALTY_INFEASIBLE,)
    nv, td = solution_cost(routes, cmap, depot)
    # BIG = 1e6 garante hierarquia: 1 veículo extra > qualquer distância em instâncias Solomon
    return (nv * 1_000_000.0 + td,)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DEAP
# ══════════════════════════════════════════════════════════════════════════════

def configurar_deap(cmap, depot, cap):
    # Evita criar múltiplas vezes em execuções repetidas
    if not hasattr(creator, "FitnessVRPTW"):
        creator.create("FitnessVRPTW", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividuoVRPTW"):
        creator.create("IndividuoVRPTW", list, fitness=creator.FitnessVRPTW)

    toolbox = base.Toolbox()

    toolbox.register("evaluate",
                     lambda ind: avaliar(ind, cmap, depot, cap))
    toolbox.register("mate",
                     lambda i1, i2: rbx_crossover(i1, i2, cmap, depot, cap))
    toolbox.register("mutate_or_opt",
                     lambda ind: mutation_or_opt(ind, cmap, depot, cap))
    toolbox.register("mutate_shuffle",
                     lambda ind: mutation_route_shuffle(ind, cmap, depot, cap))
    toolbox.register("mutate_insert",
                     lambda ind: mutation_insert_random(ind, cmap, depot, cap))
    toolbox.register("select",
                     tools.selTournament, tournsize=3)

    return toolbox


# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL DO GA
# ══════════════════════════════════════════════════════════════════════════════

def rotas_para_individuo(routes: List[List[int]]) -> "creator.IndividuoVRPTW":
    ind = creator.IndividuoVRPTW([list(r) for r in routes if r])
    return ind


def aplicar_mutacao(ind, toolbox):
    """Escolhe aleatoriamente entre os 3 operadores de mutação."""
    r = random.random()
    if r < 0.4:
        return toolbox.mutate_or_opt(ind)
    elif r < 0.7:
        return toolbox.mutate_insert(ind)
    else:
        return toolbox.mutate_shuffle(ind)


def executar_ga(clientes, deposito, capacidade, num_veiculos, cmap,
                pop_size=200, n_gen=300, cxpb=0.7, mutpb=0.3,
                elitism=5, time_limit=120.0, seed=42,
                ls_interval=10, ls_time_budget=3.0) -> Tuple[List[List[int]], float]:
    """
    GA principal com:
    - População inicial via Solomon I1 (múltiplas sementes)
    - RBX crossover
    - 3 operadores de mutação
    - Busca local periódica (Or-opt + 2-opt + relocate)
    - Elitismo
    - Critério de parada: tempo ou gerações
    """
    random.seed(seed)
    t_start = time.time()

    toolbox = configurar_deap(cmap, deposito, capacidade)

    # ── Geração da população inicial ────────────────────────────────────────
    print("Gerando população inicial (Solomon I1)...")
    raw_pop = gerar_populacao_inicial(clientes, deposito, capacidade, cmap,
                                     pop_size, seed)
    pop = [rotas_para_individuo(r) for r in raw_pop]

    # Avaliação inicial
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    best_ind = tools.selBest(pop, 1)[0]
    best_fit  = best_ind.fitness.values[0]
    nv0, td0 = solution_cost(list(best_ind), cmap, deposito)
    print(f"Melhor inicial: {nv0} veículos | dist={td0:.2f} | fit={best_fit:.2f}")

    # ── Loop evolutivo ───────────────────────────────────────────────────────
    for gen in range(1, n_gen+1):
        elapsed = time.time() - t_start
        if elapsed >= time_limit:
            print(f"Limite de tempo atingido na geração {gen}.")
            break

        # Seleção + clonagem
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # Crossover (RBX)
        for i in range(0, len(offspring)-1, 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i], offspring[i+1])

        # Mutação
        for ind in offspring:
            if random.random() < mutpb:
                aplicar_mutacao(ind, toolbox)

        # Avaliação dos inválidos
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)

        # Elitismo: preserva os melhores
        elite = tools.selBest(pop, elitism)
        pop = tools.selBest(offspring + list(map(toolbox.clone, elite)), pop_size)

        # Busca local periódica no melhor da população
        if gen % ls_interval == 0:
            melhor_atual = tools.selBest(pop, 1)[0]
            routes_ls = busca_local_completa(
                list(melhor_atual), cmap, deposito, capacidade,
                time_limit_s=ls_time_budget,
                eliminar_rotas=True
            )
            ind_ls = rotas_para_individuo(routes_ls)
            ind_ls.fitness.values = toolbox.evaluate(ind_ls)

            if ind_ls.fitness.values[0] < melhor_atual.fitness.values[0]:
                # substitui o pior da população pela solução melhorada
                worst_idx = pop.index(tools.selWorst(pop, 1)[0])
                pop[worst_idx] = ind_ls

        # Atualiza best global
        current_best = tools.selBest(pop, 1)[0]
        if current_best.fitness.values[0] < best_fit:
            best_fit  = current_best.fitness.values[0]
            best_ind  = toolbox.clone(current_best)
            nv, td = solution_cost(list(best_ind), cmap, deposito)
            print(f"  Gen {gen:4d} | veículos={nv} | dist={td:.2f} | t={elapsed:.1f}s")

    # ── Busca local final intensa ────────────────────────────────────────────
    print("\nBusca local final...")
    remaining_time = time_limit - (time.time() - t_start)
    final_ls_time = max(5.0, min(30.0, remaining_time * 0.5))
    best_routes = busca_local_completa(
        list(best_ind), cmap, deposito, capacidade,
        time_limit_s=final_ls_time,
        eliminar_rotas=True
    )

    return best_routes, best_fit


# ══════════════════════════════════════════════════════════════════════════════
# SAÍDA FORMATADA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_solucao(routes: List[List[int]], cmap, depot):
    nv, td = solution_cost(routes, cmap, depot)
    print(f"\n{'='*50}")
    print(f"MELHOR SOLUÇÃO ENCONTRADA")
    print(f"  Veículos   : {nv}")
    print(f"  Distância  : {td:.2f}")
    print(f"{'='*50}")
    for i, r in enumerate(routes, 1):
        print(f"Route #{i}: {' '.join(map(str, r))}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VRPTW — GA + Busca Local (DEAP)")
    parser.add_argument("instancia")
    parser.add_argument("--time",   type=float, default=120.0,
                        help="Limite de tempo em segundos (padrão: 120)")
    parser.add_argument("--pop",    type=int,   default=200,
                        help="Tamanho da população (padrão: 200)")
    parser.add_argument("--gen",    type=int,   default=500,
                        help="Máximo de gerações (padrão: 500)")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--ls_interval", type=int, default=10,
                        help="Aplicar busca local a cada N gerações (padrão: 10)")
    args = parser.parse_args()

    clientes, deposito, capacidade, num_veiculos = ler_instancia(args.instancia)
    cmap = {c["id"]: c for c in [deposito] + clientes}

    print(f"Instância  : {args.instancia}")
    print(f"Clientes   : {len(clientes)}")
    print(f"Capacidade : {capacidade}")
    print(f"Veículos   : {num_veiculos}")
    print(f"Tempo max  : {args.time}s | Pop: {args.pop} | Gen: {args.gen}\n")

    best_routes, best_fit = executar_ga(
        clientes, deposito, capacidade, num_veiculos, cmap,
        pop_size=args.pop,
        n_gen=args.gen,
        time_limit=args.time,
        seed=args.seed,
        ls_interval=args.ls_interval,
    )

    imprimir_solucao(best_routes, cmap, deposito)


if __name__ == "__main__":
    main()