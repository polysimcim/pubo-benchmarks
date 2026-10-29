import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


class _PairReducer:
    """
    Редукция PUBO → QUBO через переменные y_ij = x_i * x_j
    
    Для каждого произведения вводится одна вспомогательная переменная
    """
    
    def __init__(self):
        self.n_original = 0
        self.pair_to_aux = {}  # отображение (i,j) -> индекс вспомогательной
        self.aux_to_pair = {}  # отображение индекс -> (i,j)
    
    def reduce(self, H1: np.ndarray, H2: np.ndarray, H3: np.ndarray, penalty: float) -> csr_matrix:
        """
        Редукция PUBO в QUBO
        
        Args:
            H1: (N,) линейные члены
            H2: (N,N) квадратичные
            H3: (N,N,N) кубические
        
        Returns:
            Q: разреженная матрица (M x M)
        """
        self.n_original = len(H1)
        
        # Находим все пары, которые реально нужны
        needed_pairs = set()
        
        
        
        # Из кубических членов
        for i in range(self.n_original):
            for j in range(i+1, self.n_original):
                for k in range(j+1, self.n_original):
                    if abs(H3[i, j, k]) > 1e-10:
                        needed_pairs.add((i, j))
                        
        
        # Сортируем пары для детерминированного порядка
        needed_pairs = sorted(needed_pairs)
        
        # Создаем отображения
        self.pair_to_aux = {pair: idx + self.n_original 
                           for idx, pair in enumerate(needed_pairs)}
        self.aux_to_pair = {idx + self.n_original: pair 
                           for idx, pair in enumerate(needed_pairs)}
        
        n_aux = len(needed_pairs)
        n_qubo = self.n_original + n_aux
        
        print(f"Исходных переменных: {self.n_original}")
        print(f"Вспомогательных (пары): {n_aux}")
        print(f"Всего QUBO переменных: {n_qubo}")
        
        # Строим разреженную матрицу
        Q = lil_matrix((n_qubo, n_qubo), dtype=np.float64)
        b = np.zeros(n_qubo)
        
        # 1. Линейные  квадратичные члены 
        for i in range(self.n_original):
            b[i] = H1[i]
            for j in range(i+1, self.n_original):
                Q[i, j] += H2[i, j] / 2
                Q[j, i] += H2[i, j] / 2
                
                    
        for pair in self.pair_to_aux:
            
            aux = self.pair_to_aux[pair]
            i, j = pair
            
            # penalty * (x_0*x_1 - 2x_0*y - 2x_1*y - x_0 - x_1 + 2y)
            b[i] += -1 * penalty  
            b[j] += -1 * penalty  
            b[aux] += 2 * penalty  
            Q[i, j] += penalty / 2          
            Q[j, i] += penalty / 2         
            Q[i, aux] += -2 * penalty / 2  
            Q[aux, i] += -2 * penalty / 2  
            Q[j, aux] += -2 * penalty / 2     
            Q[aux, j] += -2 * penalty / 2  
        
        

        
        # 3. Кубические члены x_i * x_j * x_k = y_ij * x_k
        for i in range(self.n_original):
            for j in range(i+1, self.n_original):
                for k in range(j+1, self.n_original):
                    if abs(H3[i, j, k]) > 1e-10:
                        pair_ij = (i, j)
                        aux = self.pair_to_aux[pair_ij]
                        Q[aux, k] += 2 * H3[i, j, k] / 2
                        Q[k, aux] += 2 * H3[i, j, k] / 2
                        
                        Q[i, k] += -1 * H3[i, j, k] / 2
                        Q[k, i] += -1 * H3[i, j, k] / 2
                        Q[j, k] += -1 * H3[i, j, k] / 2
                        Q[k, j] += -1 * H3[i, j, k] / 2
                        
                        b[k] += H3[i, j, k]
                        
        
        return Q.tocsr(), b, self.pair_to_aux
    
    def recover_solution(self, qubo_sol: np.ndarray) -> np.ndarray:
        """
        Восстанавливает исходное решение
        
        Args:
            qubo_sol: решение QUBO задачи
        
        Returns:
            решение для исходных переменных
        """
        orig = qubo_sol[:self.n_original].copy()
        
        # Проверяем согласованность со вспомогательными
        for aux, pair in self.aux_to_pair.items():
            if aux < len(qubo_sol):
                i, j = pair
                y_ij = qubo_sol[aux]
                prod = orig[i] * orig[j]
                
                # Если не совпадает - корректируем (выбираем большинство)
                if abs(y_ij - prod) > 0.5:
                    if y_ij == 1:
                        # y_ij=1 значит i и j должны быть 1
                        orig[i] = 1
                        orig[j] = 1
        
        return orig


class GreedyPairReducer:
    """
    Редукция PUBO → QUBO через переменные y_ij = x_i * x_j
    
    Для каждого произведения вводится одна вспомогательная переменная
    """
    
    def __init__(self):
        self.n_original = 0
        self.pair_to_aux = {}  # отображение (i,j) -> индекс вспомогательной
        self.aux_to_pair = {}  # отображение индекс -> (i,j)
    
    def reduce(self, H1: np.ndarray, H2: np.ndarray, H3: np.ndarray, p0: float) -> csr_matrix:
        """
        Редукция PUBO в QUBO
        
        Args:
            H1: (N,) линейные члены
            H2: (N,N) квадратичные
            H3: (N,N,N) кубические
        
        Returns:
            Q: разреженная матрица (M x M)
        """
        self.n_original = len(H1)
        
        # Находим все пары, которые реально нужны
        needed_pairs = set()
        
        p_vec = np.zeros(self.n_original)
        
        # Из кубических членов
        for i in range(self.n_original):
            for j in range(i+1, self.n_original):
                for k in range(j+1, self.n_original):
                    if abs(H3[i, j, k]) > 1e-10:
                        if p_vec[i] >= p_vec[j] and p_vec[i] >= p_vec[k]:
                            
                            needed_pairs.add((j, k))
                            p_vec[j] += 1
                            p_vec[k] += 1
                        elif p_vec[j] >= p_vec[i] and p_vec[j] >= p_vec[k]:
                            needed_pairs.add((i, k))
                            p_vec[i] += 1
                            p_vec[k] += 1
                        else:
                            needed_pairs.add((i, j))
                            p_vec[i] += 1
                            p_vec[j] += 1
        
        # Сортируем пары для детерминированного порядка
        needed_pairs = sorted(needed_pairs)
        
        # Создаем отображения
        self.pair_to_aux = {pair: idx + self.n_original 
                           for idx, pair in enumerate(needed_pairs)}
        self.aux_to_pair = {idx + self.n_original: pair 
                           for idx, pair in enumerate(needed_pairs)}
        
        n_aux = len(needed_pairs)
        n_qubo = self.n_original + n_aux
        
        print(f"Исходных переменных: {self.n_original}")
        print(f"Вспомогательных (пары): {n_aux}")
        print(f"Всего QUBO переменных: {n_qubo}")
        
        # Строим разреженную матрицу
        Q = np.zeros((n_qubo, n_qubo))
        b = np.zeros(n_qubo)
        
        # 1. Линейные  квадратичные члены 
        for i in range(self.n_original):
            b[i] = H1[i]
            for j in range(i+1, self.n_original):
                Q[i, j] += H2[i, j] / 2
                Q[j, i] += H2[i, j] / 2
                
         
        # 3. Кубические члены x_i * x_j * x_k = y_ij * x_k
        for i in range(self.n_original):
            for j in range(i+1, self.n_original):
                for k in range(j+1, self.n_original):
                    if abs(H3[i, j, k]) > 1e-10:
                        try:
                            aux = self.pair_to_aux[(i, j)]
                            t = [i, j, k]
                        except:
                            try:
                                aux = self.pair_to_aux[(i, k)]
                                t = [i, k, j]
                            except:
                                aux = self.pair_to_aux[(j, k)]
                                t = [j, k, i]
                            
                        
                        Q[aux, t[2]] += 2 * H3[i, j, k] / 2
                        Q[t[2], aux] += 2 * H3[i, j, k] / 2
                        
                        Q[t[0], t[2]] += -1 * H3[i, j, k] / 2
                        Q[t[2], t[0]] += -1 * H3[i, j, k] / 2
                        Q[t[1], t[2]] += -1 * H3[i, j, k] / 2
                        Q[t[2], t[1]] += -1 * H3[i, j, k] / 2
                        
                        b[t[2]] += H3[i, j, k] * p0
                        
                        #penalty:
                        penalty = 1 * abs(H3[i, j, k])
                        # penalty * (x_0*x_1 - 2x_0*y - 2x_1*y - x_0 - x_1 + 2y)
                        b[t[0]] += -1 * penalty  
                        b[t[1]] += -1 * penalty  
                        b[aux] += 2 * penalty  
                        Q[t[0], t[1]] += penalty / 2          
                        Q[t[1], t[0]] += penalty / 2         
                        Q[t[0], aux] += -2 * penalty / 2  
                        Q[aux, t[0]] += -2 * penalty / 2  
                        Q[t[1], aux] += -2 * penalty / 2     
                        Q[aux, t[1]] += -2 * penalty / 2  
                        
        ind = [i for i in range(200, n_qubo) if np.sum(abs(Q[i])) + abs(b[i]) < 1e-10]
        Q, b = np.delete(np.delete(Q, ind, axis=0), ind, axis=1), np.delete(b, ind) 
        

        
        
                        
        
        return Q, b, self.pair_to_aux
    
    def recover_solution(self, qubo_sol: np.ndarray) -> np.ndarray:
        """
        Восстанавливает исходное решение
        
        Args:
            qubo_sol: решение QUBO задачи
        
        Returns:
            решение для исходных переменных
        """
        orig = qubo_sol[:self.n_original].copy()
        
        # Проверяем согласованность со вспомогательными
        for aux, pair in self.aux_to_pair.items():
            if aux < len(qubo_sol):
                i, j = pair
                y_ij = qubo_sol[aux]
                prod = orig[i] * orig[j]
                
                # Если не совпадает - корректируем (выбираем большинство)
                if abs(y_ij - prod) > 0.5:
                    if y_ij == 1:
                        # y_ij=1 значит i и j должны быть 1
                        orig[i] = 1
                        orig[j] = 1
        
        return orig


class PUBOToQUBOReducer:
    """PUBO -> QUBO reduction used in Sec. III C of the paper.

    One auxiliary variable is introduced for every distinct pair (i, j) formed by
    the first two indices of a cubic term J_ijk (i < j < k), with a penalty term of
    weight ``penalty`` enforcing that it equals the Boolean product of the pair.
    ``reduce(..., penalty=...)`` returns (Q as csr_matrix, linear terms, pair map).

    For backward compatibility, ``reduce(..., p0=...)`` dispatches to
    ``GreedyPairReducer`` (pairs chosen greedily, penalty |J_ijk| per term).
    """

    def __init__(self):
        self._impl = None

    def reduce(self, H1, H2, H3, penalty=None, p0=None):
        if penalty is None and p0 is None:
            raise TypeError("reduce() requires either penalty= or p0=")
        self._impl = _PairReducer() if penalty is not None else GreedyPairReducer()
        return self._impl.reduce(H1, H2, H3, penalty if penalty is not None else p0)

    def recover_solution(self, qubo_sol):
        return self._impl.recover_solution(qubo_sol)

    def __getattr__(self, name):
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)
