#!/usr/bin/env python3
# validar.py — Validador e comparador de soluções VRPTW
#
# Lê pares (.sol, .txt) da pasta Data/Solomon e executa o algoritmo
# vrptw_ga_v2.py em cada instância, comparando com o ótimo do .sol.
#
# Uso:
#   python validar.py                         # usa Data/Solomon por padrão
#   python validar.py --dir caminho/pasta     # pasta customizada
#   python validar.py --time 60 --pop 100     # parâmetros do GA
#   python validar.py --only-parse           # apenas lê .sol, sem executar GA
#
# Saída:
#   - Tabela formatada no terminal (rich)
#   - Arquivo resultados_solomon.csv

import os
import re
import sys
import math
import time
import argparse
import csv
import subprocess
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, List, Dict


def ler_sol(path_sol: str) -> Tuple[int, float, List[List[int]]]:
    """
    Retorna (num_veiculos_otimo, custo_otimo, rotas_otimas).
    Suporta linhas 'Cost X' ou 'Distance X' e variantes com '='.
    """
    rotas = []
    custo = None
    with open(path_sol, encoding="utf-8", errors="replace") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue

            # Linha de rota: "Route #N: c1 c2 ..."  ou  "Route N: c1 c2 ..."
            m = re.match(r"Route\s*#?\d+\s*:\s*(.*)", linha, re.IGNORECASE)
            if m:
                clientes = list(map(int, m.group(1).split()))
                rotas.append(clientes)
                continue

            # Linha de custo: "Cost 827.3"  ou  "Distance = 827.3"  etc.
            m = re.match(r"(?:Cost|Distance|Dist)\s*[=:]?\s*([\d.]+)", linha, re.IGNORECASE)
            if m:
                custo = float(m.group(1))

    num_veiculos = len(rotas)
    return num_veiculos, custo, rotas


# ══════════════════════════════════════════════════════════════════════════════
# LEITURA RÁPIDA DO .txt PARA METADADOS (sem rodar o GA)
# ══════════════════════════════════════════════════════════════════════════════

def ler_metadados_txt(path_txt: str) -> Tuple[int, int, int]:
    """
    Retorna (num_clientes, num_veiculos_disponiveis, capacidade).
    Leitura leve — não constrói estrutura completa.
    """
    num_clientes = 0
    capacidade = None
    num_veiculos = None
    with open(path_txt, encoding="utf-8", errors="replace") as f:
        linhas = [l.strip() for l in f if l.strip()]

    # Linha "K C": número de veículos e capacidade
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

    # Conta linhas de dados de clientes (7 campos numéricos)
    depot_found = False
    for l in linhas:
        parts = l.split()
        if len(parts) < 7:
            continue
        try:
            [float(x) for x in parts[:7]]
            if not depot_found:
                depot_found = True   # depósito (id=0)
                continue
            num_clientes += 1
        except ValueError:
            pass

    return num_clientes, num_veiculos or 0, capacidade or 0


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO DO ALGORITMO GA via subprocess
# ══════════════════════════════════════════════════════════════════════════════

def executar_ga_subprocess(path_txt: str, time_limit: float,
                           pop: int, gen: int, seed: int,
                           script: str = "vrp.py") -> Optional[Tuple[int, float]]:
    """
    Executa vrptw_ga_v2.py como subprocesso e captura a saída.
    Retorna (num_veiculos_ga, custo_ga) ou None em caso de erro.
    """
    cmd = [
        sys.executable, script,
        path_txt,
        "--time", str(time_limit),
        "--pop",  str(pop),
        "--gen",  str(gen),
        "--seed", str(seed),
    ]
    try:
        resultado = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 30,
        )
        saida = resultado.stdout + resultado.stderr
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print(f"  [ERRO] Script '{script}' não encontrado.")
        return None

    # Parseia linhas de saída do GA
    # Formato esperado na saída de imprimir_solucao():
    #   Veículos   : N
    #   Distância  : D.DD
    nv, dist = None, None
    for linha in saida.splitlines():
        m = re.search(r"Ve[íi]culos\s*:\s*(\d+)", linha, re.IGNORECASE)
        if m:
            nv = int(m.group(1))
        m = re.search(r"Dist[âa]ncia\s*:\s*([\d.]+)", linha, re.IGNORECASE)
        if m:
            dist = float(m.group(1))

    if nv is not None and dist is not None:
        return (nv, dist)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO DO ALGORITMO GA via import direto (mais rápido, sem subprocesso)
# ══════════════════════════════════════════════════════════════════════════════

def executar_ga_import(path_txt: str, time_limit: float,
                       pop: int, gen: int, seed: int,
                       script: str = "vrp.py") -> Optional[Tuple[int, float]]:
    """
    Importa vrptw_ga_v2 diretamente e executa a função executar_ga().
    Mais eficiente que subprocess (evita overhead de inicialização).
    """
    spec = importlib.util.spec_from_file_location("vrp", script)
    if spec is None:
        print(f"  [ERRO] Não foi possível carregar '{script}'.")
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [ERRO] Falha ao carregar módulo: {e}")
        return None

    try:
        clientes, deposito, capacidade, num_veiculos = mod.ler_instancia(path_txt)
        cmap = {c["id"]: c for c in [deposito] + clientes}
        best_routes, _ = mod.executar_ga(
            clientes, deposito, capacidade, num_veiculos, cmap,
            pop_size=pop,
            n_gen=gen,
            time_limit=time_limit,
            seed=seed,
        )
        nv, td = mod.solution_cost(best_routes, cmap, deposito)
        return (nv, td)
    except Exception as e:
        print(f"  [ERRO] Exceção durante execução: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DESCOBERTA DE PARES (.sol, .txt)
# ══════════════════════════════════════════════════════════════════════════════

def descobrir_pares(pasta: str) -> List[Tuple[str, str, str]]:
    """
    Retorna lista de (nome_instancia, path_sol, path_txt) ordenada.
    Emparelha arquivos com o mesmo stem (ex: C101.sol ↔ C101.txt).
    """
    pasta = Path(pasta)
    sols = {f.stem.upper(): f for f in pasta.glob("*.sol")}
    txts = {f.stem.upper(): f for f in pasta.glob("*.txt")}

    pares = []
    for stem in sorted(sols.keys()):
        if stem in txts:
            pares.append((stem, str(sols[stem]), str(txts[stem])))
        else:
            print(f"  [AVISO] {stem}.sol sem par .txt — ignorado.")

    for stem in sorted(txts.keys()):
        if stem not in sols:
            print(f"  [AVISO] {stem}.txt sem par .sol — ignorado.")

    return pares



def _fmt_float(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"

def _fmt_int(v: Optional[int]) -> str:
    return "—" if v is None else str(v)

def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sinal = "+" if v >= 0 else ""
    return f"{sinal}{v:.2f}%"


def imprimir_tabela_rich(resultados: List[Dict]):
    """Tabela colorida com rich."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    tabela = Table(
        title="[bold cyan]Comparativo VRPTW — Solomon[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )

    tabela.add_column("Instância",    style="bold", justify="left")
    tabela.add_column("Clientes",     justify="right")
    tabela.add_column("Capacidade",   justify="right")
    tabela.add_column("Veíc. Disp.", justify="right")
    tabela.add_column("Veíc. GA",    justify="right")
    tabela.add_column("Veíc. Ótimo", justify="right")
    tabela.add_column("Custo GA",    justify="right")
    tabela.add_column("Custo Ótimo", justify="right")
    tabela.add_column("Desvio %",    justify="right")

    for r in resultados:
        desvio = r.get("desvio_pct")
        if desvio is None:
            desvio_str = "[dim]—[/dim]"
            style = ""
        elif desvio <= 0.0:
            desvio_str = f"[green]{_fmt_pct(desvio)}[/green]"
            style = ""
        elif desvio <= 5.0:
            desvio_str = f"[yellow]{_fmt_pct(desvio)}[/yellow]"
            style = ""
        else:
            desvio_str = f"[red]{_fmt_pct(desvio)}[/red]"
            style = ""

        tabela.add_row(
            r["nome"],
            _fmt_int(r.get("num_clientes")),
            _fmt_int(r.get("capacidade")),
            _fmt_int(r.get("veic_disponiveis")),
            _fmt_int(r.get("veic_ga")),
            _fmt_int(r.get("veic_otimo")),
            _fmt_float(r.get("custo_ga")),
            _fmt_float(r.get("custo_otimo")),
            desvio_str,
            style=style,
        )

    console.print(tabela)

    # Resumo
    validos = [r for r in resultados if r.get("desvio_pct") is not None]
    if validos:
        media = sum(r["desvio_pct"] for r in validos) / len(validos)
        console.print(f"\n[bold]Desvio médio:[/bold] {_fmt_pct(media)}  "
                      f"[dim]({len(validos)}/{len(resultados)} instâncias executadas)[/dim]")


def imprimir_tabela_simples(resultados: List[Dict]):
    """Tabela ASCII sem dependências externas."""
    COLS = [
        ("Instância",    16, "nome"),
        ("Clientes",      8, "num_clientes"),
        ("Capac.",        7, "capacidade"),
        ("V.Disp",        6, "veic_disponiveis"),
        ("V.GA",          5, "veic_ga"),
        ("V.Ótimo",       7, "veic_otimo"),
        ("Custo GA",     10, "custo_ga"),
        ("Custo Ótimo",  12, "custo_otimo"),
        ("Desvio %",      9, "desvio_pct"),
    ]

    sep = "+" + "+".join("-" * (w + 2) for _, w, _ in COLS) + "+"
    header = "|" + "|".join(f" {h:<{w}} " for h, w, _ in COLS) + "|"
    print(sep)
    print(header)
    print(sep.replace("-", "="))

    for r in resultados:
        cells = []
        for h, w, key in COLS:
            val = r.get(key)
            if key == "desvio_pct":
                s = _fmt_pct(val)
            elif key in ("custo_ga", "custo_otimo"):
                s = _fmt_float(val)
            elif key == "nome":
                s = str(val) if val else "—"
            else:
                s = _fmt_int(val)
            cells.append(f" {s:<{w}} ")
        print("|" + "|".join(cells) + "|")
    print(sep)

    validos = [r for r in resultados if r.get("desvio_pct") is not None]
    if validos:
        media = sum(r["desvio_pct"] for r in validos) / len(validos)
        print(f"\nDesvio médio: {_fmt_pct(media)}  ({len(validos)}/{len(resultados)} instâncias)")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAÇÃO CSV
# ══════════════════════════════════════════════════════════════════════════════

def exportar_csv(resultados: List[Dict], path_out: str):
    campos = [
        "nome", "num_clientes", "capacidade", "veic_disponiveis",
        "veic_ga", "veic_otimo", "custo_ga", "custo_otimo", "desvio_pct",
    ]
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        for r in resultados:
            row = {k: ("" if r.get(k) is None else r[k]) for k in campos}
            writer.writerow(row)
    print(f"\nResultados exportados para: {path_out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validador VRPTW — compara GA com ótimos da pasta Data/Solomon"
    )
    parser.add_argument("--dir",        default="Data/Solomon",
                        help="Pasta com pares .sol/.txt (padrão: Data/Solomon)")
    parser.add_argument("--script",     default="vrp.py",
                        help="Caminho do script do GA (padrão: vrp.py)")
    parser.add_argument("--time",       type=float, default=120.0,
                        help="Limite de tempo por instância em segundos (padrão: 120)")
    parser.add_argument("--pop",        type=int,   default=200,
                        help="Tamanho da população do GA (padrão: 200)")
    parser.add_argument("--gen",        type=int,   default=500,
                        help="Máximo de gerações (padrão: 500)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--only-parse", action="store_true",
                        help="Apenas lê os .sol, sem executar o GA")
    parser.add_argument("--subprocess", action="store_true",
                        help="Executa o GA via subprocess em vez de import direto")
    parser.add_argument("--csv",        default="resultados_solomon.csv",
                        help="Arquivo de saída CSV (padrão: resultados_solomon.csv)")
    parser.add_argument("--no-rich",    action="store_true",
                        help="Desativa rich (tabela ASCII simples)")
    args = parser.parse_args()

    # Verifica pasta
    if not os.path.isdir(args.dir):
        print(f"[ERRO] Pasta '{args.dir}' não encontrada.")
        sys.exit(1)

    pares = descobrir_pares(args.dir)
    if not pares:
        print(f"[ERRO] Nenhum par .sol/.txt encontrado em '{args.dir}'.")
        sys.exit(1)

    print(f"Pasta      : {args.dir}")
    print(f"Instâncias : {len(pares)}")
    if not args.only_parse:
        print(f"Script GA  : {args.script}")
        print(f"Tempo/inst.: {args.time}s | Pop: {args.pop} | Gen: {args.gen}")
    print()

    # Decide função de execução
    if args.only_parse:
        executar_fn = None
    elif args.subprocess:
        executar_fn = lambda txt: executar_ga_subprocess(
            txt, args.time, args.pop, args.gen, args.seed, args.script)
    else:
        # Verifica se o script existe antes de importar
        if not os.path.isfile(args.script):
            print(f"[AVISO] Script '{args.script}' não encontrado. "
                  f"Use --only-parse para apenas ler .sol, "
                  f"ou --script para especificar o caminho correto.")
            executar_fn = None
        else:
            executar_fn = lambda txt: executar_ga_import(
                txt, args.time, args.pop, args.gen, args.seed, args.script)

    resultados = []
    t_total = time.time()

    for i, (nome, path_sol, path_txt) in enumerate(pares, 1):
        print(f"[{i:3d}/{len(pares)}] {nome}")

        # Lê ótimo do .sol
        try:
            veic_otimo, custo_otimo, rotas_otimas = ler_sol(path_sol)
        except Exception as e:
            print(f"         [ERRO .sol] {e}")
            veic_otimo, custo_otimo, rotas_otimas = None, None, []

        # Lê metadados do .txt
        try:
            num_clientes, veic_disp, capacidade = ler_metadados_txt(path_txt)
        except Exception as e:
            print(f"         [ERRO .txt] {e}")
            num_clientes, veic_disp, capacidade = None, None, None

        # Executa GA
        veic_ga, custo_ga = None, None
        if executar_fn is not None:
            print(f"         Executando GA...", end="", flush=True)
            t0 = time.time()
            resultado_ga = executar_fn(path_txt)
            elapsed = time.time() - t0
            if resultado_ga:
                veic_ga, custo_ga = resultado_ga
                print(f" veículos={veic_ga} | custo={custo_ga:.2f} | {elapsed:.1f}s")
            else:
                print(f" [FALHA] ({elapsed:.1f}s)")

        # Calcula desvio
        desvio_pct = None
        if custo_ga is not None and custo_otimo is not None and custo_otimo > 0:
            desvio_pct = (custo_ga - custo_otimo) / custo_otimo * 100.0

        resultados.append({
            "nome":            nome,
            "num_clientes":    num_clientes,
            "capacidade":      capacidade,
            "veic_disponiveis": veic_disp,
            "veic_ga":         veic_ga,
            "veic_otimo":      veic_otimo,
            "custo_ga":        custo_ga,
            "custo_otimo":     custo_otimo,
            "desvio_pct":      desvio_pct,
        })

    print(f"\nTempo total: {time.time()-t_total:.1f}s\n")

    # Imprime tabela
    use_rich = not args.no_rich
    if use_rich:
        try:
            imprimir_tabela_rich(resultados)
        except ImportError:
            print("[INFO] rich não instalado — usando tabela simples.")
            imprimir_tabela_simples(resultados)
    else:
        imprimir_tabela_simples(resultados)

    # Exporta CSV
    exportar_csv(resultados, args.csv)


if __name__ == "__main__":
    main()