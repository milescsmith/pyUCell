from anndata import AnnData
import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def minimal_ranker(arr: npt.NDArray, ascending: bool = False) -> int:
    arr = np.ravel(arr)
    sorter = np.argsort(arr, kind="quicksort")

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.hstack((np.array([True]), arr[1:] != arr[:-1]))

    count = np.hstack((np.nonzero(obs)[0], np.array([len(obs)])))
    dense = obs.cumsum()[inv]
    result = 0.5 * (count[dense] + count[dense - 1] + 1)

    if ascending:
        return result
    else:
        return len(result) + 1 - result


@njit
def get_ranks_of_zeros(n_genes: int, n_nonzero: int) -> npt.NDArray:
    n_zeros = n_genes - n_nonzero
    base_rank = (sum(range(n_genes + 1)) - sum(range(n_nonzero + 1))) / n_zeros
    new_arr = np.full((n_genes,), fill_value=base_rank)
    return new_arr


@njit
def insert_ranks_of_nonzero(
    base_rank_arr: npt.NDArray, nonzero_arr: npt.NDArray, indices: npt.NDArray
):
    rnk = minimal_ranker(nonzero_arr)
    base_rank_arr[indices] = rnk


@njit(parallel=True)
def rank_genes(arr: npt.NDArray) -> npt.NDArray:
    new_arr: npt.NDArray = get_ranks_of_zeros(len(arr), np.sum(arr == 0))
    indices: npt.NDArray = np.nonzero(arr)
    insert_ranks_of_nonzero(new_arr, arr[indices], indices)
    return new_arr


@np.vectorize(otypes=[int], signature="(m),()->()")
def get_index(index, label):
    return np.where(index == label)[0]


@np.vectorize(otypes=[np.float64], signature="(m),(),(),(n)->()")
def vec_calculate_u_score(cell_exprs, max_rank, n_signature, idx):
    rnk = get_ranks_of_zeros(len(cell_exprs), np.sum(cell_exprs == 0))
    indices = np.nonzero(cell_exprs)
    insert_ranks_of_nonzero(rnk, cell_exprs[indices], indices)

    rnk[rnk > max_rank] = max_rank + 1
    rnk = rnk[idx]
    u_val = sum([i - (n_signature * (n_signature + 1)) / 2 for i in rnk])
    auc = 1 - (u_val / (n_signature * max_rank))
    return auc


def score_genes_ucell(
    adata: AnnData,
    signature: list[str],
    max_rank: int = 1500,
    score_name: str = "ucell_score",
    copy: bool = False,
) -> AnnData | None:
    adata = adata.copy() if copy else adata
    n_signature = len(signature)
    idx = get_index(adata.var_names, signature)

    res = vec_calculate_u_score(
        adata.X, max_rank=max_rank, n_signature=n_signature, idx=idx
    )

    adata.obs[score_name] = res
    return adata if copy else None
