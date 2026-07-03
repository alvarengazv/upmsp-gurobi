import gurobipy as gp
from gurobipy import GRB
import math
import os
import glob


def parse_dat_file(filepath):
    """
    Lê um arquivo .dat do dataset e retorna um dicionário com todos os dados
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

def resolver_gurobi_upmsp(data, a_weight):
    """
    data: Dicionário contendo todos os dados da instância.
    a_weight: O peso 'a' da Equação 15 (varia de 0 a 1).
    """
    # Desempacotando dados (conforme o formato do dataset [6])
    N = range(data['n'])               # Conjunto de Tarefas (Jobs)
    M = range(data['m'])               # Conjunto de Máquinas
    L = range(data['o'])               # Modos de operação
    H_len = data['hl']                 # Tamanho do horizonte de tempo |H|
    H = range(H_len)                   # Horários no horizonte
    D = range(data['n_day'])           # Dias
    
    P = data['processing']             # Matriz de tempo de processamento P[i][j]
    S = data['setup']                  # Matriz de Setup S[i][j][k]
    V = data['v']                      # Fator de velocidade do modo l
    lambd = data['lambda']             # Fator de consumo de energia do modo l
    power = data['pi']                 # Potência de cada máquina i
    
    peak_s = data['peak_start']        # Início do pico por dia
    peak_e = data['peak_end']          # Fim do pico por dia
    rate_on = data['rate_in_peak']     # Tarifa de energia (pico)
    rate_off = data['rate_off_peak']   # Tarifa de energia (fora de pico)
    
    max_cost = data['max_cost']        # Limite máximo de custo para normalização
    sizeD = 24                         # Discretização do dia (ex: 24 se for em horas) [7]

    # --- Criação do Modelo ---
    model = gp.Model("UPMSP_Energy_Scheduling")
    
    # LIMITADOR DE TEMPO: 1 Hora (3600 segundos) conforme sua solicitação
    model.setParam('TimeLimit', 3600)
    
    # --- Variáveis de Decisão [1, 8] ---
    # X_ijhl: 1 se tarefa j na maq i inicia no tempo h no modo l
    X = model.addVars(M, N, H, L, vtype=GRB.BINARY, name="X")
    
    # Variáveis Contínuas (Eqs. 10, 11, 12) [3]
    Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
    TEC = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="TEC")
    PECon = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="PECon")
    PECoff = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, name="PECoff")

    # --- Restrições ---

    # Eq (3): Cada tarefa deve ser processada exatamente uma vez [1]
    for j in N:
        model.addConstr(
            gp.quicksum(X[i, j, h, l] 
                        for i in M for l in L for h in H 
                        if h <= H_len - math.ceil(P[i][j] / V[l])) == 1,
            name=f"Exec_Job_{j}"
        )

    # Eq (4): Restrição de capacidade, precedência e tempo de setup [1]
    for i in M:
        for j in N:
            for k in N:
                if j != k:
                    for l in L:
                        for h in H:
                            p_ij = math.ceil(P[i][j] / V[l])
                            limite_sup = min(h + p_ij + S[i][j][k] - 1, H_len - 1)
                            
                            model.addConstr(
                                X[i, j, h, l] + gp.quicksum(
                                    X[i, k, u, l1] 
                                    for l1 in L 
                                    for u in range(h, limite_sup + 1) if u in H
                                ) <= 1,
                                name=f"Setup_{i}_{j}_{k}_{h}_{l}"
                            )

    # Eq (5): Cálculo do Makespan (Cmax deve ser maior que o término da última tarefa) [1]
    for i in M:
        for j in N:
            for h in H:
                for l in L:
                    p_ij = math.ceil(P[i][j] / V[l])
                    model.addConstr(
                        Cmax >= X[i, j, h, l] * (h + p_ij),
                        name=f"Cmax_def_{i}_{j}_{h}_{l}"
                    )

    # Eq (6) e (7): Custos Parciais de Energia Pré-Calculados [1, 2]
    for t in D:
        expr_on = gp.LinExpr()
        expr_off = gp.LinExpr()
        
        ps = peak_s[t]
        pe = peak_e[t]
        
        for i in M:
            for j in N:
                for l in L:
                    p_ij = math.ceil(P[i][j] / V[l])
                    
                    for h in H:
                        job_end = h + p_ij
                        # Calcula a interseção do tempo da tarefa com o horário de pico
                        overlap_on = max(0, min(job_end, pe) - max(h, ps))
                        overlap_off = p_ij - overlap_on
                        
                        # Multiplicadores de custo da máquina e modo
                        fator_custo_base = lambd[l] * power[i] * (24 / sizeD)
                        
                        cost_on = overlap_on * fator_custo_base * rate_on
                        cost_off = overlap_off * fator_custo_base * rate_off
                        
                        expr_on += X[i, j, h, l] * cost_on
                        expr_off += X[i, j, h, l] * cost_off
                        
        model.addConstr(PECon[t] >= expr_on, name=f"PECon_day_{t}")
        model.addConstr(PECoff[t] >= expr_off, name=f"PECoff_day_{t}")

    # Eq (8): Custo Total de Energia [3]
    model.addConstr(TEC >= gp.quicksum(PECon[t] + PECoff[t] for t in D), name="TEC_Total")

    # --- Função Objetivo (Eq 15 - Soma Ponderada) [4] ---
    model.setObjective(
        (a_weight * (Cmax / H_len)) + ((1 - a_weight) * (TEC / max_cost)),
        GRB.MINIMIZE
    )

    # --- Otimização ---
    model.optimize()

    # Retorno de resultados (se o modelo encontrou solução ótima ou viável)
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        return {
            "a_weight": a_weight,
            "Objective": model.objVal,
            "Cmax": Cmax.X,
            "TEC": TEC.X,
            "Status": model.status
        }
    else:
        return {"a_weight": a_weight, "Status": model.status}


# --- Algoritmo 1: Simulador da Fronteira Pareto [5] ---
def run_weighted_sum_method(dados_instancia):
    pareto_front = []
    
    # Pesos definidos pelo artigo (0.0 até 1.0 com passo de 0.1) [5]
    pesos_A = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for a in pesos_A:
        print(f"\nRodando modelo para o peso a = {a}...")
        resultado = resolver_gurobi_upmsp(dados_instancia, a)
        pareto_front.append(resultado)
        
    return pareto_front


# --- Bloco Principal ---
if __name__ == "__main__":
    # Diretório com os arquivos de dados
    set1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "set1")
    dat_files = sorted(
        glob.glob(os.path.join(set1_dir, "*.dat")),
        key=lambda f: int(os.path.basename(f).split("_")[0])  # Ordena por nº de jobs
    )

    if not dat_files:
        print(f"Nenhum arquivo .dat encontrado em {set1_dir}")
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
            # Executa a fronteira de Pareto via soma ponderada
            resultados = run_weighted_sum_method(dados)

            # Imprime resumo dos resultados
            print(f"\n--- Resultados para {nome} ---")
            print(f"{'Peso a':>8} | {'Objetivo':>12} | {'Cmax':>10} | {'TEC':>12} | {'Status':>8}")
            print("-" * 62)
            for r in resultados:
                if 'Objective' in r:
                    print(f"{r['a_weight']:8.1f} | {r['Objective']:12.4f} | "
                          f"{r['Cmax']:10.2f} | {r['TEC']:12.2f} | {r['Status']:>8}")
                else:
                    print(f"{r['a_weight']:8.1f} | {'N/A':>12} | "
                          f"{'N/A':>10} | {'N/A':>12} | {r['Status']:>8}")
        except gp.GurobiError as e:
            print(f"\n  ⚠ ERRO GUROBI para {nome}: {e}")
            print("  Pulando para a próxima instância...\n")
            continue
        print()
