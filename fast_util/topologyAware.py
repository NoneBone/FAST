import networkx as nx
import numpy as np
import time

def calculate_similarity(src_values, threadNum=40, printFlag=0):
    if not isinstance(src_values, np.ndarray):
        src_values = np.array(src_values)
    
    total_size = len(src_values)
    chunk_size = total_size // threadNum
    
    if total_size % threadNum != 0:
        total_elements = chunk_size * threadNum
        src_values = src_values[:total_elements]
    
    num_chunks = threadNum
    
    matrix = src_values.reshape(num_chunks, chunk_size)
    
    all_sim = np.zeros((num_chunks, num_chunks), dtype=int)
    for i in range(num_chunks):
        matches = (matrix[i] == matrix).sum(axis=1)
        all_sim[i] = matches
    
    if printFlag:
        print("Similarity Matrix:")
        for i in range(num_chunks):
            print(f"{i:2d} : ", end="")
            for j in range(num_chunks):
                print(f"{all_sim[i, j]:2d} ", end="")
            print(" | ")
    
    return all_sim.tolist()

def getBlossomBindMode(all_sim, threadNum=40, printFlag=0):
    G = nx.Graph()
    edges = create_undirected_edges(threadNum, all_sim)
    G.add_weighted_edges_from(edges)
    if printFlag:
        t1_s = time.time()
    matching = nx.max_weight_matching(G)
    if printFlag:
        t1 = time.time() - t1_s
        print(round(t1,3))
    if printFlag:
        total_weight = sum(G[u][v]['weight'] for u, v in matching)
        print("Matching result:", end=" ")
        print(" | ".join(f"{u}-{v}({G[u][v]['weight']})" for u, v in sorted(matching)))
        print(f"Total weight: {total_weight}, Matches: {len(matching)}")

    max_index = max(max(u, v) for u, v in matching) if matching else 0
    blossomResult = np.full(threadNum, -1, dtype=int)

    for u, v in matching:
        blossomResult[u] = v
        blossomResult[v] = u

    # avoid blank match
    blankIdx1 = -1
    blankIdx2 = -1
    skipFlag = 0
    for i in range(len(blossomResult)):
        if blossomResult[i]==-1:
            if blankIdx1 == -1:
                blankIdx1 = i
                if printFlag:
                    print(blossomResult)
            else:
                blankIdx2 = i
                blossomResult[blankIdx2] = blankIdx1
                blossomResult[blankIdx1] = blankIdx2
                if printFlag: print("[Warning] blank match:{:2d},{:2d}",blankIdx1,blankIdx2)
                blankIdx1 = -1
                blankIdx2 = -1
    return blossomResult

def generate_cpu_bind_mode(numList, indexList, max_value=40):
    allocatedFlag = [0] * max_value
    cpuBindMode = []
    unmatchedList = []
    
    for i in range(len(numList)):
        if allocatedFlag[indexList[i]] == 0:
            cpuBindMode.append([numList[i], indexList[i]])
            allocatedFlag[indexList[i]] = 1
            allocatedFlag[numList[i]] = 1
        elif allocatedFlag[numList[i]] == 0:
            allocatedFlag[numList[i]] = 2
            unmatchedList.append(numList[i])
    
    print("cpuBindMode:", cpuBindMode)
    print("Unmatched numList[i]:", unmatchedList)
    return cpuBindMode, unmatchedList

def create_undirected_edges(ThreadNum, weight):
    if ThreadNum*ThreadNum != len(weight)*len(weight[0]):
        raise ValueError("ThreadNum*ThreadNum != len(weight)")
    
    edges = []
    for i in range(ThreadNum):
        for j in range(ThreadNum):
            edges.append((i, j, weight[i][j]))
    return edges