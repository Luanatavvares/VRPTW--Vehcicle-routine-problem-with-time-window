import matplotlib.pyplot as plt
import numpy as np
import os
from metricas import (
    coletar_resultados_unificados,
    calcular_metricas_por_conjunto_tipo_clientes,
    calcular_metricas_gerais,
)

# =========================================================
# Configurações
# =========================================================

plt.style.use('seaborn-v0_8-darkgrid')

CORES = {
    'C':      '#3498db',
    'R':      '#e74c3c',
    'RC':     '#2ecc71',
    'OUTROS': '#95a5a6',
}
ORDEM_TIPOS = ['C', 'RC', 'R', 'OUTROS']   # fácil → difícil

PASTA_BASE      = "graficos"
PASTA_SOLOMON   = os.path.join(PASTA_BASE, "solomon")
PASTA_HOMBERGER = os.path.join(PASTA_BASE, "homberger")
PASTA_GERAL     = os.path.join(PASTA_BASE, "geral")


def criar_pastas_graficos():
    for pasta in [PASTA_BASE, PASTA_SOLOMON, PASTA_HOMBERGER, PASTA_GERAL]:
        os.makedirs(pasta, exist_ok=True)


def _salvar(fig, caminho):
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo: {caminho}")
    plt.close(fig)


def _rotular_barras(ax, bars, fmt='{:.2f}%', offset_frac=0.015):
    """Coloca rótulo acima de cada barra com valor formatado."""
    y_max = ax.get_ylim()[1]
    offset = y_max * offset_frac
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha='center', va='bottom', fontsize=9,
        )


# =========================================================
# SOLOMON
# =========================================================

def grafico_pizza_tipos_solomon(metricas_solomon):
    tipos, qtds, cores = [], [], []
    for tipo in ORDEM_TIPOS:
        if tipo in metricas_solomon:
            tipos.append(f"Tipo {tipo}")
            qtds.append(metricas_solomon[tipo]['quantidade_instancias'])
            cores.append(CORES[tipo])
    if not qtds:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        qtds, labels=tipos, colors=cores, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(14)
    ax.set_title('Proporção de Tipos — Solomon', fontsize=16, weight='bold', pad=20)
    ax.legend(
        [f"{tipos[i]}: {qtds[i]} inst." for i in range(len(tipos))],
        loc='upper left', bbox_to_anchor=(1, 1),
    )
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_SOLOMON, 'proporcao_tipos.png'))


def grafico_barras_desvios_solomon(metricas_solomon):
    """
    Barras agrupadas: Melhor / Média / Pior desvio por tipo.
    Eixo Y começa em 0 (melhor_desvio já é ≥ 0 no metricas.py).
    """
    tipos = [t for t in ORDEM_TIPOS if t in metricas_solomon]
    if not tipos:
        return

    medias   = [metricas_solomon[t]['media_desvio_percentual'] for t in tipos]
    melhores = [metricas_solomon[t]['melhor_desvio']           for t in tipos]
    piores   = [metricas_solomon[t]['pior_desvio']             for t in tipos]

    x, larg = np.arange(len(tipos)), 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    y_teto = max(piores) * 1.15 if piores else 5
    ax.set_ylim(0, y_teto)   # ← mínimo SEMPRE 0

    for i, (vals, label, cor) in enumerate(zip(
        [melhores, medias, piores],
        ['Melhor', 'Média', 'Pior'],
        ['#2ecc71', '#3498db', '#e74c3c'],
    )):
        bars = ax.bar(x + (i - 1) * larg, vals, larg, label=label, color=cor, alpha=0.8)
        _rotular_barras(ax, bars, fmt='{:.4f}%')

    ax.set_xlabel('Tipo de Instância', fontsize=12, weight='bold')
    ax.set_ylabel('Desvio (%)', fontsize=12, weight='bold')
    ax.set_title(
        'Desvios por Tipo — Solomon  (C=fácil → R=difícil)',
        fontsize=14, weight='bold', pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'Tipo {t}' for t in tipos])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_SOLOMON, 'comparacao_desvios.png'))


def grafico_taxa_otimas_solomon(metricas_solomon):
    tipos = [t for t in ORDEM_TIPOS if t in metricas_solomon]
    if not tipos:
        return

    taxas = [metricas_solomon[t]['perc_otimo'] for t in tipos]
    cores = [CORES[t] for t in tipos]
    labels = [f'Tipo {t}' for t in tipos]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, taxas, color=cores, alpha=0.85)

    for bar, taxa in zip(bars, taxas):
        ax.text(
            taxa + 0.8, bar.get_y() + bar.get_height() / 2,
            f'{taxa:.1f}%', va='center', fontsize=11, weight='bold',
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel('Soluções Ótimas (%)', fontsize=12, weight='bold')
    ax.set_title(
        'Taxa de Soluções Ótimas por Tipo — Solomon\n(esperado: C > RC > R)',
        fontsize=13, weight='bold', pad=12,
    )
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_SOLOMON, 'taxa_otimas_por_tipo.png'))


def grafico_boxplot_solomon(resultados_solomon):
    dados, labels, cores_box = [], [], []
    for tipo in ORDEM_TIPOS:
        vals = [r['desvio_percent'] for r in resultados_solomon if r['tipo'] == tipo]
        if vals:
            dados.append(vals)
            labels.append(f'Tipo {tipo}')
            cores_box.append(CORES[tipo])
    if not dados:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    bp = ax.boxplot(dados, labels=labels, patch_artist=True, showmeans=True, meanline=True)
    for patch, cor in zip(bp['boxes'], cores_box):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)
    for el in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[el], color='black', linewidth=1.5)

    ax.set_ylim(bottom=0)   # ← mínimo SEMPRE 0
    ax.set_xlabel('Tipo de Instância', fontsize=12, weight='bold')
    ax.set_ylabel('Desvio (%)', fontsize=12, weight='bold')
    ax.set_title('Distribuição de Desvios — Solomon', fontsize=14, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_SOLOMON, 'boxplot_desvios.png'))


# =========================================================
# HOMBERGER
# =========================================================

def grafico_pizza_tipos_homberger(metricas_homberger):
    tipos, qtds, cores = [], [], []
    for tipo in ORDEM_TIPOS:
        if tipo in metricas_homberger:
            total = sum(m['quantidade_instancias'] for m in metricas_homberger[tipo].values())
            if total:
                tipos.append(f"Tipo {tipo}")
                qtds.append(total)
                cores.append(CORES[tipo])
    if not qtds:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        qtds, labels=tipos, colors=cores, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(14)
    ax.set_title('Proporção de Tipos — Homberger', fontsize=16, weight='bold', pad=20)
    ax.legend(
        [f"{tipos[i]}: {qtds[i]} inst." for i in range(len(tipos))],
        loc='upper left', bbox_to_anchor=(1, 1),
    )
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_HOMBERGER, 'proporcao_tipos.png'))


def grafico_pizza_clientes_homberger(metricas_homberger):
    todos_clientes = set()
    for d in metricas_homberger.values():
        todos_clientes.update(d.keys())
    if not todos_clientes:
        return

    clientes_list = sorted(c for c in todos_clientes if c is not None)
    qtds = []
    for c in clientes_list:
        total = sum(d[c]['quantidade_instancias'] for d in metricas_homberger.values() if c in d)
        qtds.append(total)

    cores = plt.cm.Set3(np.linspace(0, 1, len(clientes_list)))
    labels = [f"{c} clientes" for c in clientes_list]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        qtds, labels=labels, colors=cores, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(14)
    ax.set_title('Proporção por Nº de Clientes — Homberger', fontsize=16, weight='bold', pad=20)
    ax.legend(
        [f"{labels[i]}: {qtds[i]} inst." for i in range(len(labels))],
        loc='upper left', bbox_to_anchor=(1, 1),
    )
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_HOMBERGER, 'proporcao_clientes.png'))


def grafico_desvio_medio_por_clientes_homberger(metricas_homberger):
    """
    Barras + linha de tendência por nº de clientes.
    Eixo Y começa em 0 (melhor_desvio ≥ 0 garantido no metricas.py).
    """
    todos_clientes = set()
    for d in metricas_homberger.values():
        todos_clientes.update(d.keys())
    clientes_list = sorted(c for c in todos_clientes if c is not None)
    if not clientes_list:
        return

    medias, melhores, piores = [], [], []
    for c in clientes_list:
        all_m, all_min, all_max, pesos = [], [], [], []
        for d in metricas_homberger.values():
            if c in d:
                m = d[c]
                all_m.append(m['media_desvio_percentual'])
                all_min.append(m['melhor_desvio'])   # já ≥ 0
                all_max.append(m['pior_desvio'])
                pesos.append(m['quantidade_instancias'])
        pt = sum(pesos)
        medias.append(sum(v * p for v, p in zip(all_m, pesos)) / pt)
        melhores.append(min(all_min))   # pode ser 0 se houver ótimo
        piores.append(max(all_max))

    x = np.arange(len(clientes_list))
    larg = 0.25
    fig, ax = plt.subplots(figsize=(13, 7))

    y_teto = max(piores) * 1.15 if piores else 5
    ax.set_ylim(0, y_teto)   # ← mínimo SEMPRE 0

    for i, (vals, label, cor) in enumerate(zip(
        [melhores, medias, piores],
        ['Melhor', 'Média', 'Pior'],
        ['#2ecc71', '#3498db', '#e74c3c'],
    )):
        bars = ax.bar(x + (i - 1) * larg, vals, larg, label=label, color=cor, alpha=0.8)
        _rotular_barras(ax, bars, fmt='{:.4f}%')

    # Linha de tendência da média
    ax.plot(
        x, medias, color='#2c3e50', marker='o', linewidth=2,
        linestyle='--', label='Tendência média', zorder=5,
    )

    ax.set_xlabel('Número de Clientes', fontsize=12, weight='bold')
    ax.set_ylabel('Desvio (%)', fontsize=12, weight='bold')
    ax.set_title(
        'Desvio por Nº de Clientes — Homberger\n(esperado: ↑ clientes → ↑ desvio)',
        fontsize=13, weight='bold', pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in clientes_list])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_HOMBERGER, 'desvios_por_clientes.png'))


def grafico_taxa_otimas_por_clientes_homberger(metricas_homberger):
    todos_clientes = set()
    for d in metricas_homberger.values():
        todos_clientes.update(d.keys())
    clientes_list = sorted(c for c in todos_clientes if c is not None)
    if not clientes_list:
        return

    taxas = []
    for c in clientes_list:
        acertos = sum(d[c]['acertos_otimo']        for d in metricas_homberger.values() if c in d)
        total   = sum(d[c]['quantidade_instancias'] for d in metricas_homberger.values() if c in d)
        taxas.append(acertos / total * 100 if total else 0)

    cores = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(clientes_list)))
    labels = [str(c) for c in clientes_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, taxas, color=cores, alpha=0.9)
    for bar, taxa in zip(bars, taxas):
        ax.text(
            taxa + 0.8, bar.get_y() + bar.get_height() / 2,
            f'{taxa:.1f}%', va='center', fontsize=11, weight='bold',
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel('Soluções Ótimas (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Nº de Clientes', fontsize=12, weight='bold')
    ax.set_title(
        'Taxa de Soluções Ótimas por Nº de Clientes — Homberger\n(esperado: ↓ conforme clientes aumentam)',
        fontsize=12, weight='bold', pad=12,
    )
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_HOMBERGER, 'taxa_otimas_por_clientes.png'))


def grafico_barras_desvios_por_tipo_homberger(metricas_homberger):
    """
    Barras agrupadas por tipo (C/RC/R) — média ponderada entre faixas de clientes.
    Eixo Y começa em 0.
    """
    tipos = [t for t in ORDEM_TIPOS if t in metricas_homberger]
    if not tipos:
        return

    medias, melhores, piores = [], [], []
    for tipo in tipos:
        all_m, all_min, all_max, pesos = [], [], [], []
        for m in metricas_homberger[tipo].values():
            all_m.append(m['media_desvio_percentual'])
            all_min.append(m['melhor_desvio'])   # já ≥ 0
            all_max.append(m['pior_desvio'])
            pesos.append(m['quantidade_instancias'])
        pt = sum(pesos)
        medias.append(sum(v * p for v, p in zip(all_m, pesos)) / pt)
        melhores.append(min(all_min))
        piores.append(max(all_max))

    x, larg = np.arange(len(tipos)), 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    y_teto = max(piores) * 1.15 if piores else 5
    ax.set_ylim(0, y_teto)   # ← mínimo SEMPRE 0

    for i, (vals, label, cor) in enumerate(zip(
        [melhores, medias, piores],
        ['Melhor', 'Média', 'Pior'],
        ['#2ecc71', '#3498db', '#e74c3c'],
    )):
        bars = ax.bar(x + (i - 1) * larg, vals, larg, label=label, color=cor, alpha=0.8)
        _rotular_barras(ax, bars, fmt='{:.4f}%')

    ax.set_xlabel('Tipo de Instância', fontsize=12, weight='bold')
    ax.set_ylabel('Desvio (%)', fontsize=12, weight='bold')
    ax.set_title('Desvios por Tipo — Homberger', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Tipo {t}' for t in tipos])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_HOMBERGER, 'comparacao_desvios.png'))


def grafico_boxplot_homberger(resultados_homberger):
    dados, labels, cores_box = [], [], []
    for tipo in ORDEM_TIPOS:
        vals = [r['desvio_percent'] for r in resultados_homberger if r['tipo'] == tipo]
        if vals:
            dados.append(vals)
            labels.append(f'Tipo {tipo}')
            cores_box.append(CORES[tipo])
    if not dados:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    bp = ax.boxplot(dados, labels=labels, patch_artist=True, showmeans=True, meanline=True)
    for patch, cor in zip(bp['boxes'], cores_box):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)
    for el in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[el], color='black', linewidth=1.5)

    ax.set_ylim(bottom=0)   # ← mínimo SEMPRE 0
    ax.set_xlabel('Tipo de Instância', fontsize=12, weight='bold')
    ax.set_ylabel('Desvio (%)', fontsize=12, weight='bold')
    ax.set_title('Distribuição de Desvios — Homberger', fontsize=14, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_HOMBERGER, 'boxplot_desvios.png'))


# =========================================================
# GERAL
# =========================================================

def grafico_comparativo_conjuntos(metricas):
    """
    Desvio médio Solomon vs Homberger por tipo.
    Eixo Y começa em 0.
    """
    tipos = [t for t in ORDEM_TIPOS if t in metricas["Solomon"] or t in metricas["Homberger"]]
    if not tipos:
        return

    solomon_vals, homberger_vals = [], []
    for tipo in tipos:
        if tipo in metricas["Solomon"]:
            solomon_vals.append(metricas["Solomon"][tipo]['media_desvio_percentual'])
        else:
            solomon_vals.append(0.0)

        if tipo in metricas["Homberger"]:
            all_m, pesos = [], []
            for m in metricas["Homberger"][tipo].values():
                all_m.append(m['media_desvio_percentual'])
                pesos.append(m['quantidade_instancias'])
            pt = sum(pesos)
            homberger_vals.append(sum(v * p for v, p in zip(all_m, pesos)) / pt)
        else:
            homberger_vals.append(0.0)

    x, larg = np.arange(len(tipos)), 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    y_teto = max(max(solomon_vals), max(homberger_vals)) * 1.15
    ax.set_ylim(0, y_teto)   # ← mínimo SEMPRE 0

    for i, (vals, label, cor) in enumerate(zip(
        [solomon_vals, homberger_vals],
        ['Solomon', 'Homberger'],
        ['#3498db', '#e74c3c'],
    )):
        bars = ax.bar(x + (i - 0.5) * larg, vals, larg, label=label, color=cor, alpha=0.8)
        _rotular_barras(ax, bars, fmt='{:.4f}%')

    ax.set_xlabel('Tipo de Instância', fontsize=12, weight='bold')
    ax.set_ylabel('Desvio Médio (%)', fontsize=12, weight='bold')
    ax.set_title(
        'Comparação Solomon vs Homberger — Desvio Médio por Tipo',
        fontsize=13, weight='bold', pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'Tipo {t}' for t in tipos])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_GERAL, 'comparacao_solomon_homberger.png'))


def grafico_taxa_otimas_geral(metricas):
    tipos = [t for t in ORDEM_TIPOS if t in metricas["Solomon"] or t in metricas["Homberger"]]
    if not tipos:
        return

    sol_taxas, hom_taxas = [], []
    for tipo in tipos:
        sol_taxas.append(metricas["Solomon"].get(tipo, {}).get('perc_otimo', 0))
        if tipo in metricas["Homberger"]:
            acertos = sum(m['acertos_otimo']        for m in metricas["Homberger"][tipo].values())
            total   = sum(m['quantidade_instancias'] for m in metricas["Homberger"][tipo].values())
            hom_taxas.append(acertos / total * 100 if total else 0)
        else:
            hom_taxas.append(0)

    x, larg = np.arange(len(tipos)), 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (vals, label, cor) in enumerate(zip(
        [sol_taxas, hom_taxas],
        ['Solomon', 'Homberger'],
        ['#3498db', '#e74c3c'],
    )):
        bars = ax.bar(x + (i - 0.5) * larg, vals, larg, label=label, color=cor, alpha=0.8)
        _rotular_barras(ax, bars, fmt='{:.1f}%')

    ax.set_ylim(0, 115)
    ax.set_xlabel('Tipo de Instância', fontsize=12, weight='bold')
    ax.set_ylabel('Soluções Ótimas (%)', fontsize=12, weight='bold')
    ax.set_title(
        'Taxa de Soluções Ótimas — Solomon vs Homberger\n(C=fácil → R=difícil)',
        fontsize=13, weight='bold', pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'Tipo {t}' for t in tipos])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _salvar(fig, os.path.join(PASTA_GERAL, 'taxa_otimas_geral.png'))


# =========================================================
# PRINCIPAL
# =========================================================

def gerar_todos_graficos():
    print("\n" + "=" * 80)
    print("📊 GERAÇÃO DE GRÁFICOS UNIFICADOS")
    print("=" * 80 + "\n")

    criar_pastas_graficos()

    print("📥 Coletando resultados (mesma seed do validador)...")
    resultados = coletar_resultados_unificados()
    if not resultados:
        print("❌ Nenhum resultado encontrado.")
        return
    print(f"✅ {len(resultados)} resultados coletados.\n")

    resultados_solomon   = [r for r in resultados if r['conjunto'] == 'Solomon']
    resultados_homberger = [r for r in resultados if r['conjunto'] == 'Homberger']

    print("📊 Calculando métricas...")
    metricas = calcular_metricas_por_conjunto_tipo_clientes(resultados)
    print("✅ Métricas calculadas.\n")

    # Solomon
    print("🎨 Gerando gráficos Solomon...")
    if resultados_solomon and metricas["Solomon"]:
        grafico_pizza_tipos_solomon(metricas["Solomon"])
        grafico_barras_desvios_solomon(metricas["Solomon"])
        grafico_taxa_otimas_solomon(metricas["Solomon"])
        grafico_boxplot_solomon(resultados_solomon)

    # Homberger
    print("\n🎨 Gerando gráficos Homberger...")
    if resultados_homberger and metricas["Homberger"]:
        grafico_pizza_tipos_homberger(metricas["Homberger"])
        grafico_pizza_clientes_homberger(metricas["Homberger"])
        grafico_barras_desvios_por_tipo_homberger(metricas["Homberger"])
        grafico_desvio_medio_por_clientes_homberger(metricas["Homberger"])
        grafico_taxa_otimas_por_clientes_homberger(metricas["Homberger"])
        grafico_boxplot_homberger(resultados_homberger)

    # Geral
    print("\n🎨 Gerando gráficos comparativos...")
    grafico_comparativo_conjuntos(metricas)
    grafico_taxa_otimas_geral(metricas)

    print("\n" + "=" * 80)
    print("✅ TODOS OS GRÁFICOS GERADOS!")
    print("=" * 80)
    print(f"\n📂 Estrutura:")
    print(f"   {PASTA_BASE}/")
    print(f"   ├── solomon/")
    print(f"   │   ├── proporcao_tipos.png")
    print(f"   │   ├── comparacao_desvios.png")
    print(f"   │   ├── taxa_otimas_por_tipo.png")
    print(f"   │   └── boxplot_desvios.png")
    print(f"   ├── homberger/")
    print(f"   │   ├── proporcao_tipos.png")
    print(f"   │   ├── proporcao_clientes.png")
    print(f"   │   ├── comparacao_desvios.png")
    print(f"   │   ├── desvios_por_clientes.png")
    print(f"   │   ├── taxa_otimas_por_clientes.png")
    print(f"   │   └── boxplot_desvios.png")
    print(f"   └── geral/")
    print(f"       ├── comparacao_solomon_homberger.png")
    print(f"       └── taxa_otimas_geral.png\n")


if __name__ == "__main__":
    gerar_todos_graficos()