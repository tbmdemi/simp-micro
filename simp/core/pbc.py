"""
Điều kiện biên tuần hoàn (PBC) cho tối ưu hóa hình dạng SIMP.

Xây dựng ma trận chiếu PBC cho phân tích ô cơ sở tuần hoàn,
đảm bảo trường chuyển vị tuần hoàn trên các biên đối diện.

Sử dụng phương pháp null space: tìm cơ sở của không gian null
của ma trận ràng buộc, sau đó chiếu bài toán FE lên không gian
con này để loại bỏ các bậc tự do dư thừa.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve


def build_pbc(nelx: int, nely: int, nodenrs: np.ndarray):
    """Xây dựng ma trận chiếu PBC.

    Tạo ma trận thưa PBC chiếu các bậc tự do lên không gian con
    thỏa mãn điều kiện biên tuần hoàn.

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        nodenrs: Mảng (nely+1) × (nelx+1) các ID nút.

    Returns:
        Ma trận thưa (ndof, ndof_reduced) biểu diễn phép chiếu PBC.
    """
    nnx = nelx + 1
    nny = nely + 1
    ndof = 2 * nnx * nny

    # Xác định các nút biên (chỉ số 0-based)
    # Biên trái (x=0): cột 0
    left_nodes = nodenrs[:, 0] - 1  # (nny,)
    # Biên phải (x=nelx): cột cuối
    right_nodes = nodenrs[:, nelx] - 1  # (nny,)
    # Biên dưới (y=0): hàng 0
    bottom_nodes = nodenrs[0, :] - 1  # (nnx,)
    # Biên trên (y=nely): hàng cuối
    top_nodes = nodenrs[nely, :] - 1  # (nnx,)

    # Bậc tự do cho các nút biên (0-based)
    left_dofs_u = 2 * left_nodes
    left_dofs_v = 2 * left_nodes + 1
    right_dofs_u = 2 * right_nodes
    right_dofs_v = 2 * right_nodes + 1
    bottom_dofs_u = 2 * bottom_nodes
    bottom_dofs_v = 2 * bottom_nodes + 1
    top_dofs_u = 2 * top_nodes
    top_dofs_v = 2 * top_nodes + 1

    # Master: tất cả DOF trừ biên phải/biên trên (left=right, bottom=top).
    # Biên trái và biên dưới (kể cả góc dưới-trái) phải là master để các
    # slave trên biên phải/biên trên có nơi tham chiếu.
    master_mask = np.ones(ndof, dtype=bool)

    # Loại bỏ biên phải (trừ góc trên-phải)
    for i in range(nny):
        master_mask[right_dofs_u[i]] = False
        master_mask[right_dofs_v[i]] = False

    # Loại bỏ biên trên (trừ góc trên-trái)
    for i in range(nnx):
        master_mask[top_dofs_u[i]] = False
        master_mask[top_dofs_v[i]] = False

    # Chỉ góc dưới-trái không bị loại bởi vòng nào ở trên -> là master duy nhất.
    master_dofs = np.where(master_mask)[0]
    n_master = len(master_dofs)

    # Xây dựng ma trận chiếu PBC
    rows = []
    cols = []
    vals = []

    # Ánh xạ master dof -> chỉ số trong master_dofs
    master_to_idx = {dof: idx for idx, dof in enumerate(master_dofs)}

    # Master dofs: giữ nguyên
    for idx, dof in enumerate(master_dofs):
        rows.append(dof)
        cols.append(idx)
        vals.append(1.0)

    # Slave dofs: ánh xạ đến master tương ứng
    # Biên phải = biên trái
    for i in range(nny):
        # u_right = u_left
        if left_dofs_u[i] in master_to_idx:
            rows.append(right_dofs_u[i])
            cols.append(master_to_idx[left_dofs_u[i]])
            vals.append(1.0)
        # v_right = v_left
        if left_dofs_v[i] in master_to_idx:
            rows.append(right_dofs_v[i])
            cols.append(master_to_idx[left_dofs_v[i]])
            vals.append(1.0)

    # Biên trên = biên dưới
    for i in range(nnx):
        # u_top = u_bottom
        if bottom_dofs_u[i] in master_to_idx:
            rows.append(top_dofs_u[i])
            cols.append(master_to_idx[bottom_dofs_u[i]])
            vals.append(1.0)
        # v_top = v_bottom
        if bottom_dofs_v[i] in master_to_idx:
            rows.append(top_dofs_v[i])
            cols.append(master_to_idx[bottom_dofs_v[i]])
            vals.append(1.0)

    # BUG FIX: góc trên-phải không phải master lẫn không phải slave của hai
    # vòng trên (cả góc trên-trái và góc dưới-phải mà nó cần tham chiếu đều
    # đã bị loại khỏi master), nên trước fix hàng của nó trong pbc_mat toàn
    # số 0 -> chuyển vị bị ghim cứng = 0 thay vì tuần hoàn đúng. Nhờ tính bắc
    # cầu của PBC (4 góc tương đương nhau), nối thẳng nó về góc dưới-trái -
    # master duy nhất trong 4 góc.
    bottom_left_dof_u = bottom_dofs_u[0]
    bottom_left_dof_v = bottom_dofs_v[0]
    top_right_dof_u = top_dofs_u[nelx]
    top_right_dof_v = top_dofs_v[nelx]
    if bottom_left_dof_u in master_to_idx:
        rows.append(top_right_dof_u)
        cols.append(master_to_idx[bottom_left_dof_u])
        vals.append(1.0)
    if bottom_left_dof_v in master_to_idx:
        rows.append(top_right_dof_v)
        cols.append(master_to_idx[bottom_left_dof_v])
        vals.append(1.0)

    pbc_mat = coo_matrix(
        (vals, (rows, cols)),
        shape=(ndof, n_master),
    ).tocsr()

    return pbc_mat
