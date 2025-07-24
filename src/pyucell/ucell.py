import warnings

import numpy as np
import numpy.typing as npt
from anndata import AnnData
from numba import float64, guvectorize, int64, njit


@njit(parallel=True)
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


@njit(parallel=True)
def insert_ranks_of_nonzero(
    base_rank_arr: npt.NDArray, nonzero_arr: npt.NDArray, indices: npt.NDArray
) -> None:
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


@guvectorize(
    [(float64[:], int64, int64, int64[:], float64[:])],
    "(m),(),(),(n)->()",
    nopython=True,
    nogil=True,
    cache=True,
)
def guvec_calculate_u_score(cell_exprs: npt.NDArray[np.float64], max_rank: int, n_signature: int, idx: npt.NDArray[np.int64], auc: npt.NDArray[np.float64]) -> None:
    rnk = get_ranks_of_zeros(len(cell_exprs), np.sum(cell_exprs == 0))
    indices = np.nonzero(cell_exprs)
    insert_ranks_of_nonzero(rnk, cell_exprs[indices], indices)

    rnk[rnk > max_rank] = max_rank + 1
    rnk = rnk[idx]
    u_val = sum([i - (n_signature * (n_signature + 1)) / 2 for i in rnk])

    auc[0] = 1 - (u_val / (n_signature * max_rank))


def score_genes_ucell(
    adata: AnnData,
    signature: list[str],
    max_rank: int = 1500,
    score_name: str = "ucell_score",
    copy: bool = False,
) -> AnnData | None:
    adata = adata.copy() if copy else adata
    n_signature = len(signature)
    found_signature_genes = adata.var_names.intersection(signature)
    if len(found_signature_genes) > 0:
        idx = get_index(adata.var_names, found_signature_genes)

        res = guvec_calculate_u_score(
            adata.X, max_rank=max_rank, n_signature=n_signature, idx=idx
        )

        adata.obs[score_name] = res
    else:
        warnings.warn("None of the signature genes was found. No score caluclated.", stacklevel=1)
    return adata if copy else None
