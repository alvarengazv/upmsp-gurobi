import gurobipy as gp
from gurobipy import GRB
import math
import os
import glob


def parse_dat_file(filepath):
    """
    Lê um arquivo .txt do dataset e retorna um dicionário com todos os dados
    da instância, no formato esperado por resolver_gurobi_upmsp().
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
            vals, idx = read_vector(idx, data['n_day'])
            data['peak_start'] = [int(v) for v in vals]

        elif line == 'peak_end':
            vals, idx = read_vector(idx, data['n_day'])
            data['peak_end'] = [int(v) for v in vals]

        elif line == 'v':
            vals, idx = read_vector(idx, data['o'])
            data['v'] = vals

        elif line == 'lambda':
            vals, idx = read_vector(idx, data['o'])
            data['lambda'] = vals

        elif line == 'pi':
            vals, idx = read_vector(idx, data['m'])
            data['pi'] = vals

        elif line == 'processing':
            # Arquivo: n linhas × m colunas (job × máquina)
            # Modelo espera: P[i][j] = P[máquina][job] → transpor
            mat, idx = read_matrix(idx, data['n'], data['m'])
            # Transpõe para m×n (máquina × job) e converte para int
            data['processing'] = [[int(mat[j][i]) for j in range(data['n'])]
                                  for i in range(data['m'])]

        elif line == 'setup':
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
    Executa o modelo Gurobi para uma única instância e um único peso 'a'.
    Versão OTIMIZADA: reduz horizonte de tempo, pré-computa tempos de processamento
    e usa addConstrs em lote para evitar exaustão de memória RAM.

    data: Dicionário contendo todos os dados da instância lidos do arquivo de texto.
    a_weight: O peso 'a' da Equação 15. Padrão é 0.5 (peso igual para Tempo e Energia).
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
    # Evita milhões de chamadas redundantes a math.ceil dentro dos geradores.
    # =========================================================================
    proc = {}  # proc[i, j, l] = ceil(P[i][j] / V[l])
    for i in M:
        for j in N:
            for l in L:
                proc[i, j, l] = math.ceil(P[i][j] / V[l])

    # =========================================================================
    # OTIMIZAÇÃO 2: Redução dinâmica do Horizonte de Tempo (H)
    # Em vez de usar hl=1439 cegamente, calculamos o pior makespan possível
    # (soma dos piores tempos de processamento + piores setups) e limitamos H.
    # Para 6 tarefas isso reduz H de 1439 para ~450, eliminando ~70% das vars.
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

    # --- Variáveis de Decisão ---
    # Só criamos variáveis para h viáveis (h <= H_len - proc[i,j,l])
    print("  Criando variáveis de decisão (somente para h viáveis)...")
    var_keys = [
        (i, j, h, l)
        for i in M for j in N for l in L
        for h in range(H_len - proc[i, j, l] + 1)
    ]
    X = model.addVars(var_keys, vtype=GRB.BINARY, name="X")
    print(f"  → {len(var_keys)} variáveis X criadas "
          f"(vs. {m*n*H_len*o} sem filtragem).")

    Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
    TEC = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="TEC")
    PECon = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="PECon")
    PECoff = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="PECoff")

    # =========================================================================
    # OTIMIZAÇÃO 3: Criação de restrições em lote com addConstrs + geradores
    # =========================================================================

    # --- Eq (3): Cada tarefa deve ser processada exatamente uma vez ---
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
    print("  Construindo restrições de Setup (Eq 4) — pode demorar...")
    # Para cada (i, j, h, l) existente, nenhuma outra tarefa k pode iniciar
    # no intervalo [h, h + proc[i,j,l] + S[i][j][k] - 1] na mesma máquina.
    # Usamos gerador para evitar materializar tudo na memória.
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
    print(f"  → {setup_count} restrições de Setup criadas.")

    # --- Eq (5): Cálculo do Makespan ---
    print("  Construindo restrições de Makespan (Eq 5)...")
    model.addConstrs(
        (Cmax >= X[i, j, h, l] * (h + proc[i, j, l])
         for (i, j, h, l) in var_keys),
        name="Cmax_def"
    )

    # --- Eq (6) e (7): Custos Parciais de Energia (pico/fora-de-pico) ---
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
    model.addConstr(
        TEC >= gp.quicksum(PECon[t] + PECoff[t] for t in D),
        name="TEC_Total"
    )

    # --- Função Objetivo (Eq 15 — Soma Ponderada) ---
    model.setObjective(
        (a_weight * (Cmax / H_len)) + ((1 - a_weight) * (TEC / max_cost)),
        GRB.MINIMIZE
    )

    # =========================================================================
    # Resolver
    # =========================================================================
    print("\n  Modelo construído. Iniciando solver Gurobi...\n")
    model.optimize()

    # --- Verificação e Resultados ---
    if model.status == GRB.OPTIMAL:
        print("\nSolução ótima encontrada!")
    elif model.status == GRB.TIME_LIMIT:
        print("\nLimite de tempo atingido. Retornando melhor solução viável.")
    elif model.status == GRB.INFEASIBLE:
        print("\nO modelo é inviável.")
        return None

    if model.solCount > 0:
        # Extrai o escalonamento final para debug
        print("\n--- Escalonamento Final ---")
        for (i, j, h, l) in var_keys:
            if X[i, j, h, l].X > 0.5:
                print(f"  Máquina {i} | Job {j} | Início h={h} | "
                      f"Modo {l} (v={V[l]}) | Duração={proc[i,j,l]}")
        return {
            "Status": model.status,
            "Funcao_Objetivo": model.objVal,
            "Makespan_Final (Cmax)": Cmax.X,
            "Custo_Energia_Final (TEC)": TEC.X
        }
    else:
        return {"Status": "Sem solução no tempo limite."}


# --- Bloco Principal ---
if __name__ == "__main__":
    # Diretório com os arquivos de dados
    set1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "set1")
    dat_files = sorted(
        glob.glob(os.path.join(set1_dir, "*.txt")),
        key=lambda f: int(os.path.basename(f).split("_")[0])  # Ordena por nº de jobs
    )

    if not dat_files:
        print(f"Nenhum arquivo .txt encontrado em {set1_dir}")
        exit(1)

    print(f"Encontrados {len(dat_files)} arquivos de dados em set1/:\n")
    for f in dat_files:
        print(f"  - {os.path.basename(f)}")

    for filepath in dat_files:
        nome = os.path.basename(filepath)
        print("\n" + "=" * 70)
        print(f"  INSTÂNCIA: {nome}")
        print("=" * 70)

        # Lê e faz o parse do arquivo
        dados = parse_dat_file(filepath)
        print(f"  n={dados['n']} tarefas, m={dados['m']} máquinas, "
              f"o={dados['o']} modos, hl={dados['hl']} períodos")

        try:
            # Executa o solver para instância única com peso fixo
            resultado = resolver_gurobi_instancia_unica(dados, a_weight=0.5)

            # Imprime resumo dos resultados
            print(f"\n--- Resultados para {nome} ---")
            if resultado is not None:
                for chave, valor in resultado.items():
                    print(f"  {chave}: {valor}")
            else:
                print("  Nenhuma solução viável encontrada.")
        except gp.GurobiError as e:
            print(f"\n  ⚠ ERRO GUROBI para {nome}: {e}")
            print("  Pulando para a próxima instância...\n")
            continue
        print()

