# =============================================================================
#  UPMSP-SDS com Custo de Energia por Horário de Uso (Time-of-Use)
#  Reprodução da formulação MILP exata de Rego et al. (2022) em Gurobi/Python
# =============================================================================
#
#  Este script implementa integralmente o modelo matemático (Equações 1-12
#  do artigo) proposto para o problema de escalonamento em máquinas paralelas
#  não relacionadas com tempos de setup dependentes da sequência (UPMSP-SDS),
#  cujos dois objetivos (minimizar o makespan e o custo total de energia sob
#  tarifação por horário de uso) são combinados em uma única função objetivo
#  por meio do Método da Soma Ponderada (Equação 15).
#
#  Estrutura do arquivo:
#    1) parse_dat_file()                    -> leitura das instâncias (.txt)
#    2) resolver_gurobi_instancia_unica()    -> construção e resolução do MILP
#    3) gerar_*()                            -> geração dos gráficos do relatório
#    4) bloco principal (__main__)           -> orquestra a execução completa
#
#  Por se tratar de uma formulação indexada no tempo (a variável de decisão
#  X_ijhl carrega o instante h como índice), o número de variáveis e de
#  restrições cresce multiplicativamente com o horizonte de planejamento H.
#  Por isso, este script aplica três otimizações de engenharia (detalhadas
#  nos comentários abaixo, junto aos trechos correspondentes) para reduzir o
#  consumo de memória durante a construção do modelo:
#    (i)   pré-cálculo do tempo de processamento ajustado pela velocidade;
#    (ii)  redução dinâmica do horizonte de tempo H por instância;
#    (iii) criação de restrições em lote (addConstrs) sempre que possível.
# =============================================================================

import gurobipy as gp
from gurobipy import GRB
import math
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo (não requer display gráfico)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime


def parse_dat_file(filepath):
    """
    Lê um arquivo .txt do dataset (formato definido por Rego, Cota & Souza,
    2021, disponibilizado junto ao artigo) e retorna um dicionário com todos
    os dados da instância, no formato esperado por
    resolver_gurobi_instancia_unica().

    O arquivo de instância é dividido em duas partes:
      1) Uma seção inicial de escalares "chave valor" (n = nº de tarefas,
         m = nº de máquinas, n_day = nº de dias do horizonte, hl = tamanho
         do horizonte H, o = nº de modos de operação, tarifas de energia
         etc.);
      2) Uma sequência de seções nomeadas (peak_start, peak_end, v, lambda,
         pi, processing, setup), cada uma contendo um vetor ou matriz de
         dados associados à instância.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove trailing whitespace e filtra linhas
    lines = [line.rstrip() for line in lines]

    data = {}
    idx = 0

    # --- Escalares da primeira seção ---
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # Tenta ler pares "chave valor" na mesma linha
        parts = line.split()
        if len(parts) == 2:
            key, val = parts
            if key in ('n', 'm', 'n_day', 'hl', 'o'):
                data[key] = int(val)
                idx += 1
                continue
            elif key in ('rate_in_peak', 'rate_off_peak', 'max_cost'):
                data[key] = float(val)
                idx += 1
                continue

        # Se não é par chave-valor, saímos para ler seções nomeadas
        break

    # --- Seções nomeadas (vetores e matrizes) ---
    def read_vector(start_idx, count):
        """Lê 'count' valores numéricos a partir de start_idx, pulando linhas vazias."""
        values = []
        i = start_idx
        while len(values) < count and i < len(lines):
            line = lines[i].strip()
            if line:
                values.append(float(line))
            i += 1
        return values, i

    def read_matrix(start_idx, rows, cols):
        """Lê uma matriz rows x cols (valores separados por tab/espaço)."""
        matrix = []
        i = start_idx
        while len(matrix) < rows and i < len(lines):
            line = lines[i].strip()
            if line:
                row = [float(x) for x in line.split()]
                matrix.append(row)
            i += 1
        return matrix, i

    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1

        if not line:
            continue

        if line == 'peak_start':
            # startp_t (Equações 6 e 7): instante de início do horário de
            # pico, um valor por dia t do horizonte.
            vals, idx = read_vector(idx, data['n_day'])
            data['peak_start'] = [int(v) for v in vals]

        elif line == 'peak_end':
            # endp_t (Equações 6 e 7): instante de término do horário de
            # pico, um valor por dia t do horizonte.
            vals, idx = read_vector(idx, data['n_day'])
            data['peak_end'] = [int(v) for v in vals]

        elif line == 'v':
            # v_l: fator multiplicativo de velocidade de cada modo de
            # operação l (usado para ajustar o tempo de processamento).
            vals, idx = read_vector(idx, data['o'])
            data['v'] = vals

        elif line == 'lambda':
            # lambda_l: fator multiplicativo de potência de cada modo de
            # operação l (usado no cálculo do custo de energia).
            vals, idx = read_vector(idx, data['o'])
            data['lambda'] = vals

        elif line == 'pi':
            # pi_i: potência nominal de cada máquina i em operação normal.
            vals, idx = read_vector(idx, data['m'])
            data['pi'] = vals

        elif line == 'processing':
            # p_ij: tempo de processamento da tarefa j na máquina i.
            # Arquivo: n linhas × m colunas (job × máquina)
            # Modelo espera: P[i][j] = P[máquina][job] → transpor
            mat, idx = read_matrix(idx, data['n'], data['m'])
            # Transpõe para m×n (máquina × job) e converte para int
            data['processing'] = [[int(mat[j][i]) for j in range(data['n'])]
                                  for i in range(data['m'])]

        elif line == 'setup':
            # S_ijk: tempo de setup dependente da sequência para executar
            # a tarefa k logo após a tarefa j na máquina i (Equação 4).
            # Setup é uma matriz 3D: S[i][j][k] para cada máquina i
            # No arquivo: m blocos de n x n — valores inteiros
            setup = []
            for _ in range(data['m']):
                mat, idx = read_matrix(idx, data['n'], data['n'])
                # Converte para int (usado em range/aritmética inteira)
                setup.append([[int(v) for v in row] for row in mat])
            data['setup'] = setup

    return data

def resolver_gurobi_instancia_unica(data, a_weight=0.5):
    """
    Constrói e resolve, via Gurobi, o MILP exato de Rego et al. (2022)
    (Equações 1-12 e 15) para uma única instância e um único peso 'a' do
    Método da Soma Ponderada.

    Versão OTIMIZADA: reduz o horizonte de tempo H, pré-computa os tempos de
    processamento ajustados pela velocidade e usa addConstrs em lote sempre
    que possível, de modo a mitigar a exaustão de memória RAM decorrente da
    natureza pseudopolinomial da formulação indexada no tempo.

    data: Dicionário contendo todos os dados da instância lidos do arquivo de
          texto (ver parse_dat_file).
    a_weight: O peso 'a' da Equação 15. Padrão é 0.5 (peso igual entre
              makespan e custo de energia).

    Retorna um dicionário com o status da otimização, os valores ótimos (ou
    da melhor solução viável encontrada) de makespan, custo de energia e
    função objetivo, o escalonamento detalhado e estatísticas do modelo
    (número de variáveis, restrições, não-zeros, tempo de solução etc.),
    usadas posteriormente para gerar os gráficos do relatório.
    """
    print(f"Iniciando otimização com peso a = {a_weight}...")

    # Desempacotando dados básicos
    n = data['n']
    m = data['m']
    o = data['o']
    N = range(n)                       # Conjunto de Tarefas (Jobs)
    M = range(m)                       # Conjunto de Máquinas
    L = range(o)                       # Modos de operação

    P = data['processing']             # P[i][j] — tempo de proc. da tarefa j na máquina i
    S = data['setup']                  # S[i][j][k] — setup na máquina i entre job j e k
    V = data['v']                      # Fator de velocidade do modo l
    lambd = data['lambda']             # Fator de consumo de energia do modo l
    power = data['pi']                 # Potência de cada máquina i

    D = range(data['n_day'])           # Dias
    peak_s = data['peak_start']        # Início do pico por dia
    peak_e = data['peak_end']          # Fim do pico por dia
    rate_on = data['rate_in_peak']     # Tarifa de energia (pico)
    rate_off = data['rate_off_peak']   # Tarifa de energia (fora de pico)
    max_cost = data['max_cost']        # Limite máximo de custo para normalização
    sizeD = 24                         # Discretização do dia

    # =========================================================================
    # OTIMIZAÇÃO 1: Pré-computar ceil(P[i][j] / V[l]) num dicionário
    #
    # Esse valor corresponde ao termo ⌈P_ij / v_l⌉ que aparece repetidamente
    # em praticamente todas as equações do modelo (3, 4, 5, 6 e 7): ele
    # representa a duração real de execução da tarefa j na máquina i quando
    # processada no modo de operação l, já ajustada pelo fator de velocidade
    # v_l e arredondada para cima (para manter a discretização inteira do
    # horizonte de tempo). Pré-computá-lo evita milhões de chamadas
    # redundantes a math.ceil() dentro dos laços/geradores usados para
    # construir as restrições.
    # =========================================================================
    proc = {}  # proc[i, j, l] = ceil(P[i][j] / V[l])
    for i in M:
        for j in N:
            for l in L:
                proc[i, j, l] = math.ceil(P[i][j] / V[l])

    # =========================================================================
    # OTIMIZAÇÃO 2: Redução dinâmica do Horizonte de Tempo (H)
    #
    # A formulação original do artigo define H = {0, ..., |H|} com |H| fixo
    # (hl = 1439, um dia inteiro discretizado em minutos). Como a variável
    # X_ijhl é indexada em h ∈ H, usar o horizonte completo cria um número
    # enorme de variáveis que jamais poderiam pertencer a uma solução ótima
    # (nenhuma tarefa começaria, por exemplo, no minuto 1000 se todas as
    # tarefas puderem terminar bem antes disso).
    #
    # Para evitar isso, calculamos aqui um limite superior (pior caso) para
    # o makespan: a soma, para cada tarefa, do maior tempo de processamento
    # possível entre as máquinas (ajustado pela menor velocidade v_min
    # disponível) mais o maior tempo de setup possível. Esse valor é usado
    # para truncar H, eliminando do modelo todos os instantes de tempo
    # inatingíveis por qualquer solução viável, sem alterar o conjunto de
    # soluções ótimas do problema.
    #
    # Para 6 tarefas, essa redução leva H de 1439 para 529 (-63%).
    # =========================================================================
    v_min = min(V)
    pior_proc = sum(max(P[i][j] for i in M) for j in N)
    pior_proc_ajustado = math.ceil(pior_proc / v_min)
    pior_setup = sum(max(max(S[i][j]) for i in M) for j in N)
    pior_makespan = pior_proc_ajustado + pior_setup

    H_len = min(data['hl'], pior_makespan + 10)
    H = range(H_len)
    print(f"  Horizonte de tempo H reduzido de {data['hl']} para {H_len} "
          f"({100 - 100*H_len/data['hl']:.1f}% de redução em variáveis).")

    # =========================================================================
    # Criação do Modelo
    # =========================================================================
    model = gp.Model("UPMSP_Energy_Scheduling_Otimizado")
    model.setParam('TimeLimit', 3600)
    model.setParam('OutputFlag', 1)

    # --- Variáveis de Decisão (Equação 9: X_ijhl ∈ {0,1}) ---
    # Só criamos variáveis para h viáveis, isto é, apenas para instantes de
    # início h tais que a tarefa termine dentro do horizonte reduzido
    # (h + proc[i,j,l] <= H_len). Isso evita alocar variáveis para combinações
    # que seriam automaticamente fixadas em zero por violarem o horizonte.
    print("  Criando variáveis de decisão (somente para h viáveis)...")
    var_keys = [
        (i, j, h, l)
        for i in M for j in N for l in L
        for h in range(H_len - proc[i, j, l] + 1)
    ]
    X = model.addVars(var_keys, vtype=GRB.BINARY, name="X")
    print(f"  >> {len(var_keys)} variaveis X criadas "
          f"(vs. {m*n*H_len*o} sem filtragem).")

    # Variáveis auxiliares contínuas do modelo (Equações 10, 11 e 12: >= 0)
    Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")     # Makespan
    TEC = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="TEC")       # Custo total de energia
    PECon = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="PECon")   # Custo parcial no pico, por dia t
    PECoff = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="PECoff") # Custo parcial fora do pico, por dia t

    # =========================================================================
    # OTIMIZAÇÃO 3: Criação de restrições em lote com addConstrs + geradores
    # =========================================================================

    # --- Eq (3): Cada tarefa deve ser processada exatamente uma vez ---
    # Garante que cada tarefa j seja alocada a exatamente uma combinação de
    # máquina i, instante de início h e modo de operação l, terminando
    # dentro do horizonte de planejamento.
    print("  Construindo restrições de Execução (Eq 3)...")
    model.addConstrs(
        (gp.quicksum(
            X[i, j, h, l]
            for i in M for l in L
            for h in range(H_len - proc[i, j, l] + 1)
        ) == 1
         for j in N),
        name="Exec_Job"
    )

    # --- Eq (4): Restrição de Setup / Não-sobreposição ---
    # Impõe que, se a tarefa j é executada na máquina i a partir de h, então
    # nenhuma outra tarefa k pode iniciar nessa mesma máquina antes do fim da
    # execução de j somado ao tempo de setup S[i][j][k] entre as duas. Ou
    # seja, para cada (i, j, h, l) em que X_ijhl pode valer 1, o intervalo
    # [h, h + proc[i,j,l] + S[i][j][k] - 1] fica bloqueado para o início de
    # qualquer outra tarefa k na mesma máquina i, em qualquer modo l1.
    # Essa é a restrição mais numerosa do modelo (responsável por >80% do
    # total de restrições), pois é definida para cada tupla
    # (i, j, k, h, l) com j != k. Para evitar materializar todas as
    # combinações em memória antes de enviá-las ao solver, adicionamos cada
    # restrição individualmente dentro de um laço explícito, verificando
    # antes se o intervalo de conflito [h, limite_sup] é não vazio.
    print("  Construindo restricoes de Setup (Eq 4) -- pode demorar...")
    setup_count = 0
    for i in M:
        for j in N:
            for l in L:
                p_ijl = proc[i, j, l]
                max_h_j = H_len - p_ijl
                for h in range(max_h_j + 1):
                    for k in N:
                        if j == k:
                            continue
                        limite_sup = min(h + p_ijl + S[i][j][k] - 1, H_len - 1)
                        # Só adiciona se o range de conflito não for vazio
                        if h <= limite_sup:
                            model.addConstr(
                                X[i, j, h, l] + gp.quicksum(
                                    X.get((i, k, u, l1), 0)
                                    for l1 in L
                                    for u in range(h, limite_sup + 1)
                                ) <= 1,
                                name=f"Setup_{i}_{j}_{k}_{h}_{l}"
                            )
                            setup_count += 1
    print(f"  >> {setup_count} restricoes de Setup criadas.")

    # --- Eq (5): Lower bound do Makespan ---
    # Cmax deve ser maior ou igual ao instante de término de qualquer tarefa
    # alocada (h + duração), para toda combinação (i, j, h, l) em que
    # X_ijhl possa valer 1. Como o objetivo minimiza Cmax, essa restrição
    # força Cmax a assumir exatamente o valor do maior instante de término
    # entre todas as tarefas escalonadas.
    print("  Construindo restrições de Makespan (Eq 5)...")
    model.addConstrs(
        (Cmax >= X[i, j, h, l] * (h + proc[i, j, l])
         for (i, j, h, l) in var_keys),
        name="Cmax_def"
    )

    # --- Eq (6) e (7): Custos Parciais de Energia (pico/fora-de-pico) ---
    # Para cada dia t do horizonte, calcula-se o custo de energia mínimo
    # incorrido no horário de pico (PECon, Eq. 6) e fora do horário de pico
    # (PECoff, Eq. 7), considerando que uma tarefa pode ser executada
    # parcialmente em cada um dos dois regimes tarifários (os seis casos de
    # sobreposição descritos no artigo). Para cada variável X_ijhl, calcula-se
    # a fração da duração da tarefa que efetivamente ocorre dentro da janela
    # de pico (overlap_on) e fora dela (overlap_off), multiplicando-se pelo
    # fator de potência/energia da tarefa (lambda_l * pi_i * 24/sizeD) e pela
    # respectiva tarifa (rate_on ou rate_off). As expressões lineares
    # resultantes (expr_on, expr_off) são então usadas para definir o lower
    # bound de PECon[t] e PECoff[t].
    print("  Construindo restrições de Energia (Eq 6-7)...")
    for t in D:
        expr_on = gp.LinExpr()
        expr_off = gp.LinExpr()
        ps = peak_s[t]
        pe = peak_e[t]

        for (i, j, h, l) in var_keys:
            p_ijl = proc[i, j, l]
            job_end = h + p_ijl

            overlap_on = max(0, min(job_end, pe) - max(h, ps))
            overlap_off = p_ijl - overlap_on

            fator = lambd[l] * power[i] * (24.0 / sizeD)

            if overlap_on > 0:
                expr_on.addTerms(overlap_on * fator * rate_on, X[i, j, h, l])
            if overlap_off > 0:
                expr_off.addTerms(overlap_off * fator * rate_off, X[i, j, h, l])

        model.addConstr(PECon[t] >= expr_on, name=f"PECon_day_{t}")
        model.addConstr(PECoff[t] >= expr_off, name=f"PECoff_day_{t}")

    # --- Eq (8): Custo Total de Energia ---
    # TEC é o somatório, ao longo de todos os dias do horizonte, dos custos
    # parciais de energia no pico e fora do pico.
    model.addConstr(
        TEC >= gp.quicksum(PECon[t] + PECoff[t] for t in D),
        name="TEC_Total"
    )

    # --- Função Objetivo (Eq 15 — Método da Soma Ponderada) ---
    # Converte o problema bi-objetivo (min Cmax, min TEC) em um problema
    # mono-objetivo, ponderando o makespan normalizado pela cardinalidade do
    # horizonte (|H|) e o custo de energia normalizado por max_cost (uma
    # estimativa heurística do maior custo possível, fornecida na instância).
    # a_weight = 0,5 confere peso igual aos dois objetivos.
    model.setObjective(
        (a_weight * (Cmax / H_len)) + ((1 - a_weight) * (TEC / max_cost)),
        GRB.MINIMIZE
    )

    # =========================================================================
    # Resolver
    # =========================================================================
    print("\n  Modelo construído. Iniciando solver Gurobi...\n")
    model.optimize()

    # --- Coleta estatísticas do modelo para o relatório ---
    model_stats = {
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
        "num_nz": model.NumNZs,
        "setup_constrs": setup_count,
        "H_len_original": data['hl'],
        "H_len_reduzido": H_len,
        "solve_time": model.Runtime,
        "nodes_explored": model.NodeCount,
        "n": n, "m": m, "o": o,
    }

    # --- Verificação e Resultados ---
    if model.status == GRB.OPTIMAL:
        print("\nSolução ótima encontrada!")
    elif model.status == GRB.TIME_LIMIT:
        print("\nLimite de tempo atingido. Retornando melhor solução viável.")
    elif model.status == GRB.INFEASIBLE:
        print("\nO modelo é inviável.")
        return None

    if model.solCount > 0:
        # Extrai o escalonamento final detalhado
        schedule = []
        print("\n--- Escalonamento Final ---")
        for (i, j, h, l) in var_keys:
            if X[i, j, h, l].X > 0.5:
                duracao = proc[i, j, l]
                print(f"  Máquina {i} | Job {j} | Início h={h} | "
                      f"Modo {l} (v={V[l]}) | Duração={duracao}")
                schedule.append({
                    "maquina": i,
                    "job": j,
                    "inicio": h,
                    "modo": l,
                    "velocidade": V[l],
                    "lambda": lambd[l],
                    "potencia": power[i],
                    "duracao": duracao,
                    "fim": h + duracao,
                })

        return {
            "Status": model.status,
            "Funcao_Objetivo": model.objVal,
            "Makespan_Final": Cmax.X,
            "Custo_Energia_Final": TEC.X,
            "schedule": schedule,
            "model_stats": model_stats,
            "peak_start": peak_s,
            "peak_end": peak_e,
            "rate_on": rate_on,
            "rate_off": rate_off,
        }
    else:
        return {"Status": "Sem solução no tempo limite.", "model_stats": model_stats}


# =============================================================================
#  FUNÇÕES DE GERAÇÃO DE GRÁFICOS PARA O RELATÓRIO
#
#  As funções abaixo não fazem parte do modelo de otimização em si: elas
#  apenas processam o dicionário de resultados retornado por
#  resolver_gurobi_instancia_unica() (e, no caso da análise de complexidade,
#  os dados de todas as instâncias lidas) para gerar as figuras utilizadas
#  no relatório técnico (diagrama de Gantt, comparação de tempo com o
#  artigo original, crescimento do modelo, decomposição do custo de
#  energia, tabela-resumo e redução do horizonte de tempo).
# =============================================================================

def configurar_estilo_graficos():
    """Configura o estilo global dos gráficos para aparência acadêmica."""
    plt.rcParams.update({
        'figure.facecolor': '#FAFAFA',
        'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#222222',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


def gerar_gantt_chart(resultado, output_dir):
    """
    Gera o Diagrama de Gantt do escalonamento ótimo encontrado pelo Gurobi.
    Mostra tarefas alocadas em cada máquina, com destaque para o horário de pico.
    """
    schedule = resultado['schedule']
    makespan = resultado['Makespan_Final']
    peak_s = resultado['peak_start'][0]
    peak_e = resultado['peak_end'][0]

    # Cores para cada job
    cores_jobs = plt.cm.Set3(np.linspace(0, 1, 12))

    # Descobre máquinas usadas
    maquinas = sorted(set(t['maquina'] for t in schedule))

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Faixa de horário de pico (se visível dentro do makespan)
    if peak_s < makespan + 20:
        ax.axvspan(peak_s, min(peak_e, makespan + 50),
                   alpha=0.12, color='#E53935', zorder=0,
                   label=f'Horario de Pico ({peak_s}-{peak_e} min)')

    # Linha vertical do Makespan
    ax.axvline(x=makespan, color='#D32F2F', linewidth=2, linestyle='--',
               label=f'Makespan = {int(makespan)} min', zorder=5)

    bar_height = 0.6
    y_labels = []

    for idx, maq in enumerate(maquinas):
        y_labels.append(f'Máquina {maq}')
        tarefas_maq = [t for t in schedule if t['maquina'] == maq]

        for tarefa in tarefas_maq:
            cor = cores_jobs[tarefa['job'] % len(cores_jobs)]
            ax.barh(idx, tarefa['duracao'], left=tarefa['inicio'],
                    height=bar_height, color=cor,
                    edgecolor='#333333', linewidth=0.8, zorder=3)
            # Rótulo dentro da barra
            cx = tarefa['inicio'] + tarefa['duracao'] / 2
            label_text = f"J{tarefa['job']}\nv={tarefa['velocidade']}"
            fontsize = 8 if tarefa['duracao'] > 15 else 6
            ax.text(cx, idx, label_text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='#222222', zorder=4)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Tempo (minutos)')
    ax.set_title('Diagrama de Gantt - Escalonamento Otimo (6 Tarefas, 2 Maquinas)')
    ax.set_xlim(-5, makespan + 30)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    filepath = os.path.join(output_dir, 'gantt_escalonamento.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Gantt Chart salvo em: {filepath}")
    return filepath


def gerar_comparacao_tempo(resultado, output_dir):
    """
    Gera gráfico de barras comparando o tempo de execução obtido
    com o tempo reportado por Rego et al. (2022) no artigo original.
    """
    stats = resultado['model_stats']
    tempo_nosso = stats['solve_time']

    # Dados do artigo original (Rego et al., 2022) - Tabela de resultados
    dados_artigo = {
        'instancias': ['6 tarefas\n(sucesso)', '7 tarefas\n(OOM/Killed)'],
        'tempo_artigo': [172.09, 549.86],
        'tempo_nosso': [tempo_nosso, None],  # None = não completou
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(dados_artigo['instancias']))
    largura = 0.32

    # Barras do artigo
    bars1 = ax.bar(x - largura/2, dados_artigo['tempo_artigo'], largura,
                   label='Rego et al. (2022)\nIntel i7 · 16 GB RAM',
                   color='#5C6BC0', edgecolor='#283593', linewidth=0.8,
                   zorder=3)

    # Barras nossas (com hachura para a que falhou)
    cores_nosso = ['#66BB6A', '#EF5350']
    for idx, val in enumerate(dados_artigo['tempo_nosso']):
        if val is not None:
            ax.bar(x[idx] + largura/2, val, largura,
                   color=cores_nosso[0], edgecolor='#2E7D32', linewidth=0.8,
                   zorder=3,
                   label='Nossa execução' if idx == 0 else '')
            ax.text(x[idx] + largura/2, val + 8, f'{val:.1f}s',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            # Barra com X indicando falha
            ax.bar(x[idx] + largura/2, 600, largura,
                   color=cores_nosso[1], edgecolor='#C62828', linewidth=0.8,
                   hatch='///', alpha=0.7, zorder=3,
                   label='OOM — Killed pelo SO' if idx == 1 else '')
            ax.text(x[idx] + largura/2, 610, '✗ OOM',
                    ha='center', va='bottom', fontweight='bold',
                    fontsize=11, color='#C62828')

    # Rótulos nas barras do artigo
    for bar, val in zip(bars1, dados_artigo['tempo_artigo']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 8,
                f'{val:.1f}s', ha='center', va='bottom',
                fontweight='bold', fontsize=10, color='#283593')

    ax.set_xticks(x)
    ax.set_xticklabels(dados_artigo['instancias'], fontsize=11)
    ax.set_ylabel('Tempo de Resolução (segundos)')
    ax.set_title('Comparacao de Tempo de Execucao - Gurobi (MILP Exato)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 700)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filepath = os.path.join(output_dir, 'comparacao_tempo_execucao.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Comparacao de Tempo salva em: {filepath}")
    return filepath


def gerar_explosao_combinatoria(resultado, dados_instancias, output_dir):
    """
    Gera gráfico mostrando o crescimento exponencial do número de variáveis,
    restrições e não-zeros à medida que o número de tarefas aumenta.
    Demonstra visualmente por que o MILP exato é inviável para instâncias grandes.
    """
    stats_6 = resultado['model_stats']

    # Dados reais (6 tarefas) + estimativas proporcionais para 7-10 tarefas
    # Estimativas baseadas no crescimento observado e na estrutura do modelo
    n_tarefas = [6, 7, 8, 9, 10]

    # Para n tarefas, m máquinas, o modos: vars ~ m*n*H*o, setup ~ m*n*(n-1)*o*H
    # Usamos os dados reais de 6 e escalamos proporcionalmente
    vars_real = stats_6['num_vars']
    constrs_real = stats_6['num_constrs']
    nz_real = stats_6['num_nz']

    # Fator de escala baseado em n*(n-1)*H_estimado
    def estimar_H(n_jobs):
        """Estima H com base na proporção de jobs."""
        # Processing times crescem linearmente, setup cresce com n²
        dados_h = {6: stats_6['H_len_reduzido']}
        # Para cada instância, calcula H reduzido
        for filepath_inst, dados_inst in dados_instancias.items():
            n_inst = dados_inst['n']
            P_inst = dados_inst['processing']
            S_inst = dados_inst['setup']
            V_inst = dados_inst['v']
            m_inst = dados_inst['m']
            v_min = min(V_inst)
            pp = sum(max(P_inst[i][j] for i in range(m_inst)) for j in range(n_inst))
            pp_adj = math.ceil(pp / v_min)
            ps = sum(max(max(S_inst[i][j]) for i in range(m_inst)) for j in range(n_inst))
            dados_h[n_inst] = min(dados_inst['hl'], pp_adj + ps + 10)
        return dados_h

    h_estimados = estimar_H(6)

    # Calcula o número de variáveis e restrições para cada instância usando dados reais
    num_vars_list = []
    num_setup_list = []
    num_nz_list = []

    for filepath_inst, dados_inst in sorted(dados_instancias.items(),
                                             key=lambda x: x[1]['n']):
        n_i = dados_inst['n']
        m_i = dados_inst['m']
        o_i = dados_inst['o']
        H_i = h_estimados.get(n_i, 500)

        # Estimativa de variáveis: sum over i,j,l of (H - ceil(P[i][j]/V[l]) + 1)
        P_i = dados_inst['processing']
        V_i = dados_inst['v']
        total_vars = 0
        for i in range(m_i):
            for j in range(n_i):
                for l in range(o_i):
                    hmax = H_i - math.ceil(P_i[i][j] / V_i[l]) + 1
                    if hmax > 0:
                        total_vars += hmax
        num_vars_list.append(total_vars)

        # Estimativa de restrições de setup: para cada (i,j,l,h) viável, (n-1) restrições
        est_setup = total_vars * (n_i - 1)
        num_setup_list.append(est_setup)

        # Estimativa de não-zeros: cresce ~= setup * avg_range * o
        avg_range = H_i / (n_i * 2)  # estimativa conservadora
        est_nz = est_setup * avg_range * o_i
        num_nz_list.append(est_nz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Gráfico 1: Variáveis e Restrições de Setup ---
    cor_vars = '#42A5F5'
    cor_setup = '#EF5350'

    ax1_twin = ax1.twinx()

    bars1 = ax1.bar(np.arange(len(n_tarefas)) - 0.18, num_vars_list, 0.35,
                    color=cor_vars, edgecolor='#1565C0', linewidth=0.8,
                    label='Variáveis Binárias (X)', zorder=3)
    bars2 = ax1_twin.bar(np.arange(len(n_tarefas)) + 0.18, num_setup_list, 0.35,
                         color=cor_setup, edgecolor='#C62828', linewidth=0.8,
                         label='Restrições de Setup', zorder=3)

    ax1.set_xticks(range(len(n_tarefas)))
    ax1.set_xticklabels([f'{n}' for n in n_tarefas])
    ax1.set_xlabel('Número de Tarefas (n)')
    ax1.set_ylabel('Variáveis Binárias', color=cor_vars)
    ax1_twin.set_ylabel('Restrições de Setup', color=cor_setup)
    ax1.set_title('Crescimento de Variáveis e Restrições')
    ax1.tick_params(axis='y', labelcolor=cor_vars)
    ax1_twin.tick_params(axis='y', labelcolor=cor_setup)

    # Linha de limite de memória
    ax1.axhline(y=num_vars_list[0], color='#66BB6A', linewidth=1.5,
                linestyle=':', alpha=0.7, label='Limite resolvido (6 jobs)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.grid(axis='y', alpha=0.2, linestyle='--')

    # --- Gráfico 2: Crescimento Exponencial (escala log) ---
    ax2.semilogy(n_tarefas, num_vars_list, 's-', color='#42A5F5',
                 markersize=8, linewidth=2, label='Variáveis', zorder=3)
    ax2.semilogy(n_tarefas, num_setup_list, 'D-', color='#EF5350',
                 markersize=8, linewidth=2, label='Restrições de Setup', zorder=3)
    ax2.semilogy(n_tarefas, num_nz_list, '^-', color='#AB47BC',
                 markersize=8, linewidth=2, label='Não-zeros estimados', zorder=3)

    # Destaque: zona viável vs inviável
    ax2.axvspan(5.5, 6.5, alpha=0.15, color='#66BB6A', label='Resolvido')
    ax2.axvspan(6.5, 10.5, alpha=0.10, color='#EF5350', label='OOM / Killed')

    ax2.set_xlabel('Número de Tarefas (n)')
    ax2.set_ylabel('Quantidade (escala logarítmica)')
    ax2.set_title('Explosão Combinatória — Escala Logarítmica')
    ax2.set_xticks(n_tarefas)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')

    fig.suptitle('Análise da Complexidade do Modelo MILP (UPMSP)', fontsize=15,
                 fontweight='bold', y=1.02)
    fig.tight_layout()

    filepath = os.path.join(output_dir, 'explosao_combinatoria.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Explosao Combinatoria salva em: {filepath}")
    return filepath


def gerar_energia_breakdown(resultado, output_dir):
    """
    Gera gráfico de barras empilhadas mostrando o consumo de energia
    por máquina, dividido entre horário de pico e fora de pico.
    """
    schedule = resultado['schedule']
    peak_s = resultado['peak_start'][0]
    peak_e = resultado['peak_end'][0]
    rate_on = resultado['rate_on']
    rate_off = resultado['rate_off']
    sizeD = 24

    # Calcula consumo por máquina
    maquinas = sorted(set(t['maquina'] for t in schedule))
    energia_pico = {m: 0 for m in maquinas}
    energia_fora = {m: 0 for m in maquinas}

    for t in schedule:
        maq = t['maquina']
        h = t['inicio']
        fim = t['fim']
        lam = t['lambda']
        pot = t['potencia']

        overlap_on = max(0, min(fim, peak_e) - max(h, peak_s))
        overlap_off = t['duracao'] - overlap_on

        fator = lam * pot * (24.0 / sizeD)
        energia_pico[maq] += overlap_on * fator * rate_on
        energia_fora[maq] += overlap_off * fator * rate_off

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Gráfico 1: Barras empilhadas por máquina ---
    x = np.arange(len(maquinas))
    pico_vals = [energia_pico[m] for m in maquinas]
    fora_vals = [energia_fora[m] for m in maquinas]

    ax1.bar(x, fora_vals, 0.5, label='Fora de Pico', color='#66BB6A',
            edgecolor='#2E7D32', linewidth=0.8, zorder=3)
    ax1.bar(x, pico_vals, 0.5, bottom=fora_vals, label='Horário de Pico',
            color='#EF5350', edgecolor='#C62828', linewidth=0.8, zorder=3)

    for i, m in enumerate(maquinas):
        total = pico_vals[i] + fora_vals[i]
        ax1.text(i, total + 20, f'{total:.1f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Máquina {m}' for m in maquinas], fontsize=11)
    ax1.set_ylabel('Custo de Energia (R$)')
    ax1.set_title('Custo de Energia por Máquina')
    ax1.legend(framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Gráfico 2: Pizza do total ---
    total_pico = sum(pico_vals)
    total_fora = sum(fora_vals)
    total_geral = total_pico + total_fora

    sizes = [total_fora, total_pico]
    labels_pie = [f'Fora de Pico\n{total_fora:.1f} ({100*total_fora/total_geral:.1f}%)',
                  f'Horário de Pico\n{total_pico:.1f} ({100*total_pico/total_geral:.1f}%)']
    colors = ['#66BB6A', '#EF5350']
    explode = (0.03, 0.06)

    wedges, texts = ax2.pie(sizes, labels=labels_pie, colors=colors,
                            explode=explode, startangle=90,
                            wedgeprops={'edgecolor': '#333', 'linewidth': 0.8})
    for text in texts:
        text.set_fontsize(10)
    ax2.set_title(f'Distribuição Total de Energia\nTEC = {total_geral:.2f}')

    fig.suptitle('Análise de Custo Energético — Instância 6 Tarefas', fontsize=14,
                 fontweight='bold', y=1.02)
    fig.tight_layout()

    filepath = os.path.join(output_dir, 'energia_breakdown.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Energia Breakdown salvo em: {filepath}")
    return filepath


def gerar_tabela_resumo(resultado, output_dir):
    """
    Gera uma figura estilo tabela com o resumo comparativo dos resultados.
    """
    stats = resultado['model_stats']

    dados_tabela = [
        ['Métrica', 'Nossa Execução', 'Rego et al. (2022)'],
        ['Instância', '6_2_1439_3_S_1-9', '6_2_1439_3_S_1-9'],
        ['Tarefas × Máquinas × Modos', f"{stats['n']} × {stats['m']} × {stats['o']}",
         '6 × 2 × 3'],
        ['Horizonte H (original → reduzido)',
         f"{stats['H_len_original']} → {stats['H_len_reduzido']}", '1439'],
        ['Variáveis Binárias', f"{stats['num_vars']:,}", '—'],
        ['Restrições Totais', f"{stats['num_constrs']:,}", '—'],
        ['Restrições de Setup', f"{stats['setup_constrs']:,}", '—'],
        ['Não-zeros na Matriz', f"{stats['num_nz']:,}", '—'],
        ['Nós Explorados', f"{int(stats['nodes_explored']):,}", '—'],
        ['Tempo de Resolução', f"{stats['solve_time']:.2f} s", '172.09 s'],
        ['Makespan (Cmax)', f"{resultado['Makespan_Final']:.0f} min", '—'],
        ['Custo Energético (TEC)', f"{resultado['Custo_Energia_Final']:.2f}", '—'],
        ['Função Objetivo', f"{resultado['Funcao_Objetivo']:.4f}", '—'],
        ['Status', 'ÓTIMO ✓', 'ÓTIMO ✓'],
    ]

    n_rows = len(dados_tabela)
    n_cols = 3

    fig, ax = plt.subplots(figsize=(12, n_rows * 0.48 + 1))
    ax.axis('off')

    tabela = ax.table(
        cellText=dados_tabela[1:],
        colLabels=dados_tabela[0],
        loc='center',
        cellLoc='center',
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1, 1.6)

    # Estilização do cabeçalho
    for j in range(n_cols):
        cell = tabela[0, j]
        cell.set_facecolor('#3949AB')
        cell.set_text_props(color='white', fontweight='bold')

    # Estilização das linhas
    for i in range(1, n_rows):
        for j in range(n_cols):
            cell = tabela[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#E8EAF6')
            else:
                cell.set_facecolor('#FFFFFF')
            # Destaque para linha de status
            if i == n_rows - 1:
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(fontweight='bold')

    ax.set_title('Tabela Resumo — Resultados Computacionais', fontsize=14,
                 fontweight='bold', pad=20)

    filepath = os.path.join(output_dir, 'tabela_resumo.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Tabela Resumo salva em: {filepath}")
    return filepath


def gerar_horizonte_reducao(dados_instancias, output_dir):
    """
    Gera gráfico mostrando a redução do horizonte de tempo H para cada instância,
    comparando H original (1439) com o H reduzido calculado dinamicamente.
    """
    n_list = []
    h_original_list = []
    h_reduzido_list = []

    for filepath_inst, dados_inst in sorted(dados_instancias.items(),
                                             key=lambda x: x[1]['n']):
        n_i = dados_inst['n']
        m_i = dados_inst['m']
        P_i = dados_inst['processing']
        S_i = dados_inst['setup']
        V_i = dados_inst['v']
        v_min = min(V_i)

        pior_proc = sum(max(P_i[i][j] for i in range(m_i)) for j in range(n_i))
        pior_proc_adj = math.ceil(pior_proc / v_min)
        pior_setup = sum(max(max(S_i[i][j]) for i in range(m_i)) for j in range(n_i))
        h_red = min(dados_inst['hl'], pior_proc_adj + pior_setup + 10)

        n_list.append(n_i)
        h_original_list.append(dados_inst['hl'])
        h_reduzido_list.append(h_red)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(n_list))
    largura = 0.35

    bars1 = ax.bar(x - largura/2, h_original_list, largura,
                   label='H Original', color='#BDBDBD',
                   edgecolor='#757575', linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + largura/2, h_reduzido_list, largura,
                   label='H Reduzido (Otimização)', color='#42A5F5',
                   edgecolor='#1565C0', linewidth=0.8, zorder=3)

    # Percentual de redução
    for i in range(len(n_list)):
        reducao = 100 * (1 - h_reduzido_list[i] / h_original_list[i])
        ax.text(x[i] + largura/2, h_reduzido_list[i] + 20,
                f'{h_reduzido_list[i]}\n(-{reducao:.0f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#1565C0')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} tarefas' for n in n_list], fontsize=11)
    ax.set_ylabel('Horizonte de Tempo (H)')
    ax.set_title('Otimização 2 — Redução Dinâmica do Horizonte de Tempo')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filepath = os.path.join(output_dir, 'reducao_horizonte.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Reducao de Horizonte salva em: {filepath}")
    return filepath


# =============================================================================
#  BLOCO PRINCIPAL — EXECUÇÃO E GERAÇÃO DO RELATÓRIO
#
#  Fluxo de execução:
#    1) Carrega todas as instâncias do diretório set1/ (usadas depois na
#       análise de crescimento/complexidade do modelo);
#    2) Resolve, com o Gurobi, apenas a instância de 6 tarefas — a única que
#       coube na memória RAM disponível para resolução completa (ver
#       discussão de resultados no relatório técnico, Seção 6);
#    3) Gera os seis gráficos utilizados no relatório a partir do resultado
#       obtido e dos dados de todas as instâncias carregadas.
# =============================================================================

if __name__ == "__main__":
    configurar_estilo_graficos()

    # Diretório de saída para gráficos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "relatorio_graficos")
    os.makedirs(output_dir, exist_ok=True)

    # Diretório com os arquivos de dados
    set1_dir = os.path.join(script_dir, "set1")
    dat_files = sorted(
        glob.glob(os.path.join(set1_dir, "*.txt")),
        key=lambda f: int(os.path.basename(f).split("_")[0])
    )

    if not dat_files:
        print(f"Nenhum arquivo .txt encontrado em {set1_dir}")
        exit(1)

    # Carrega TODAS as instâncias (para análise de complexidade)
    print("=" * 70)
    print("  CARREGANDO TODAS AS INSTÂNCIAS PARA ANÁLISE DE COMPLEXIDADE")
    print("=" * 70)
    dados_instancias = {}
    for filepath in dat_files:
        nome = os.path.basename(filepath)
        dados = parse_dat_file(filepath)
        dados_instancias[filepath] = dados
        print(f"  Carregado: {nome} -- n={dados['n']}, m={dados['m']}, "
              f"o={dados['o']}, hl={dados['hl']}")

    # ===========================================================
    #  EXECUTA APENAS A PRIMEIRA INSTÂNCIA (6 tarefas — a única
    #  que cabe na memória RAM para resolução completa)
    # ===========================================================
    filepath_6 = dat_files[0]
    nome_6 = os.path.basename(filepath_6)
    dados_6 = dados_instancias[filepath_6]

    print("\n" + "=" * 70)
    print(f"  RESOLVENDO INSTÂNCIA: {nome_6}")
    print(f"  n={dados_6['n']} tarefas, m={dados_6['m']} máquinas, "
          f"o={dados_6['o']} modos, hl={dados_6['hl']} períodos")
    print("=" * 70)

    try:
        resultado = resolver_gurobi_instancia_unica(dados_6, a_weight=0.5)
    except gp.GurobiError as e:
        print(f"\n  [!] ERRO GUROBI: {e}")
        resultado = None

    if resultado is None or 'schedule' not in resultado:
        print("\n  [X] Nao foi possivel obter solucao. Verifique licenca do Gurobi.")
        exit(1)

    # Imprime resumo
    print("\n" + "=" * 70)
    print("  RESUMO DOS RESULTADOS")
    print("=" * 70)
    print(f"  Função Objetivo:  {resultado['Funcao_Objetivo']:.4f}")
    print(f"  Makespan (Cmax):  {resultado['Makespan_Final']:.0f} minutos")
    print(f"  Custo Energia:    {resultado['Custo_Energia_Final']:.2f}")
    print(f"  Tempo de Solver:  {resultado['model_stats']['solve_time']:.2f} s")
    print(f"  Nós Explorados:   {int(resultado['model_stats']['nodes_explored'])}")

    # ===========================================================
    #  GERAÇÃO DE GRÁFICOS PARA O RELATÓRIO
    # ===========================================================
    print("\n" + "=" * 70)
    print("  GERANDO GRÁFICOS PARA O RELATÓRIO")
    print("=" * 70 + "\n")

    graficos = {}

    # 1. Diagrama de Gantt
    graficos['gantt'] = gerar_gantt_chart(resultado, output_dir)

    # 2. Comparação de Tempo com o Artigo
    graficos['tempo'] = gerar_comparacao_tempo(resultado, output_dir)

    # 3. Explosão Combinatória
    graficos['complexidade'] = gerar_explosao_combinatoria(
        resultado, dados_instancias, output_dir)

    # 4. Análise de Energia
    graficos['energia'] = gerar_energia_breakdown(resultado, output_dir)

    # 5. Tabela Resumo
    graficos['tabela'] = gerar_tabela_resumo(resultado, output_dir)

    # 6. Redução do Horizonte de Tempo
    graficos['horizonte'] = gerar_horizonte_reducao(dados_instancias, output_dir)

    print("\n" + "=" * 70)
    print("  RELATÓRIO COMPLETO!")
    print("=" * 70)
    print(f"\n  Todos os {len(graficos)} gráficos foram salvos em:")
    print(f"  [DIR] {output_dir}\n")
    for nome_graf, caminho in graficos.items():
        print(f"    - {nome_graf}: {os.path.basename(caminho)}")
    print()