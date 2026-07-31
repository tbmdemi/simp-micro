"""
Cập nhật theo tiêu chí tối ưu (OC) cho tối ưu hóa hình dạng SIMP.

Thực hiện thuật toán cập nhật OC (Optimality Criteria) cổ điển
để cập nhật biến thiết kế dựa trên độ nhạy và ràng buộc thể tích.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .filter import apply_heaviside_projection

# X_MIN: sàn dưới của biến thiết kế x - khớp giá trị chuẩn tham chiếu Sigmund
# (2001) 99-line MATLAB (`xnew = max(0.001, ...)`), KHÔNG phải 0.0. Lý do:
# dc/dx (dQ) ∝ x^(penal-1) (xem homogenization/compute.py) -> TẠI x=0 CHÍNH
# XÁC, độ nhạy = 0 (với penal>1), khiến x*ratio giữ nguyên 0 MÃI MÃI (cập
# nhật nhân, không phải cộng) - một trạng thái "hấp thụ" toán học không thể
# thoát ra dù có thêm bất kỳ ràng buộc/phạt nào khác (đã kiểm chứng: bug FIX
# 2026-07-29 dùng x_min=0.0 khiến `hexagonal`/`reentrant_bowtie` sụp về vol~0
# vĩnh viễn chỉ trong ~10 vòng lặp đầu, xem EXPERIMENT_LOG.md). Dùng 0.001
# giữ độ nhạy khác 0 tại "gần-void", cho phép phần tử hồi phục nếu cần.
X_MIN = 0.001


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    move: float,
    H,
    Hs,
    ft: int,
    Q: np.ndarray | None = None,
    delta: float | None = None,
    use_sqrt: bool = False,
    projection: str | None = None,
    beta_proj: float = 8.0,
    eta_proj: float = 0.5,
    stiffness_mode: str = 'gate',
    dg: np.ndarray | None = None,
):
    """Cập nhật biến thiết kế dùng tiêu chí tối ưu (OC).

    Thực hiện cập nhật OC với tìm kiếm nhị phân trên hệ số Lagrange
    để thỏa mãn ràng buộc thể tích.

    Hỗ trợ thêm ràng buộc stiffness (Q₁₁ ≥ δ, Q₂₂ ≥ δ) dùng cho auxetic objective
    (giống MATLAB topK_Hourglass.m).

    Args:
        x: Mảng (nely, nelx) biến thiết kế hiện tại.
        dc: Mảng (nely, nelx) độ nhạy hàm mục tiêu.
        dv: Mảng (nely, nelx) độ nhạy thể tích.
        volfrac: Tỉ lệ thể tích yêu cầu.
        move: Giới hạn thay đổi cho phép mỗi vòng lặp.
        H: Ma trận lọc thưa.
        Hs: Vector tổng trọng số lọc.
        ft: Loại bộ lọc (1=độ nhạy, 2=mật độ).
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3, optional). Dùng khi có ràng buộc stiffness.
        delta: Ngưỡng stiffness tối thiểu (optional). Yêu cầu Q nếu delta được cung cấp.
        use_sqrt: Nếu True, dùng x * sqrt(-dc/(dv*lmid)) (Sigmund 2001 heuristic).
                   Nếu False, dùng x * (-dc/(dv*lmid)) (MATLAB reference).
                   Mặc định False để khớp MATLAB.
        projection: None (mặc định) hoặc 'heaviside'. Nếu 'heaviside', ràng buộc
            thể tích trong bisection nhắm vào mean(x̂) = mean(apply_heaviside_
            projection(xPhys, beta_proj, eta_proj)), không phải mean(xPhys) thô
            (xPhys ở đây là x̃, trường đã lọc nhưng chưa qua projection).
        beta_proj, eta_proj: Tham số Heaviside projection (chỉ dùng khi
            projection='heaviside'), xem core/filter.py::apply_heaviside_projection().
        stiffness_mode: 'gate' (mặc định, hành vi cũ) hoặc 'dual' (TÍNH NĂNG
            THỬ NGHIỆM, TẮT MẶC ĐỊNH - xem oc_update_dual()). 'gate' AND một
            điều kiện nhị phân stiff_ok vào bisection thể tích duy nhất - Q
            được evaluate MỘT LẦN tại x cũ và giữ NGUYÊN suốt cả 100 vòng
            bisection nhị phân bên dưới, nên khi vi phạm (stiff_ok=False),
            điều kiện `vol > volfrac and stiff_ok` luôn False bất kể lmid,
            bisection sụp về lmid->0 (l1 không bao giờ được set) - TOÀN BỘ
            ràng buộc thể tích bị bỏ qua trong vòng lặp đó, mọi phần tử có
            -dc>0 bị đẩy lên x+move ĐỒNG LOẠT (không theo trọng số độ nhạy),
            gây thể tích vọt lên; vòng sau stiff_ok=True trở lại (thể tích dư
            thừa), constraint đổi lại thành thuần volume, bisection kéo mật
            độ xuống mạnh - dao động qua lại (limit-cycle). Đây là nguyên
            nhân xác nhận (không phải giả thuyết) khiến tăng delta/beta ở
            EXPERIMENT_LOG.md mục 2026-07-29 làm dao động BÙNG NỔ thay vì cải
            thiện: tăng delta khiến stiff_ok=False xảy ra thường xuyên hơn,
            kích hoạt cơ chế "bỏ ràng buộc thể tích + đẩy đồng loạt" này
            thường xuyên hơn.
        dg: Mảng (nely, nelx) độ nhạy CỦA THÀNH PHẦN STIFFNESS RÀNG BUỘC CHẶT
            HƠN (min(Q11,Q22)) theo x - chỉ dùng khi stiffness_mode='dual'.
            Caller (runner.py) tự chọn dQ[binding,binding] và áp cùng chuỗi
            projection-derivative + sensitivity-filter như dc trước khi
            truyền vào (xem oc_update_dual() để biết công thức đầy đủ).

    Returns:
        Bộ (xnew, xPhys) với:
            xnew : Mảng (nely, nelx) biến thiết kế mới (chưa lọc).
            xPhys: Mảng (nely, nelx) mật độ x̃ đã lọc (CHƯA projection - caller
                tự áp projection để có x̂, giống hành vi cũ).
    """
    nely, nelx = x.shape

    # Xác định có ràng buộc stiffness hay không
    has_stiffness_constraint = (Q is not None) and (delta is not None)

    if stiffness_mode == 'dual' and has_stiffness_constraint:
        if dg is None:
            raise ValueError("stiffness_mode='dual' yêu cầu tham số dg (xem docstring).")
        return oc_update_dual(
            x, dc, dv, dg, volfrac, delta, move, H, Hs, ft, Q,
            use_sqrt=use_sqrt, projection=projection,
            beta_proj=beta_proj, eta_proj=eta_proj,
        )

    l1 = 0.0
    l2 = 1e9

    # Tìm kiếm nhị phân cho hệ số Lagrange
    # Lặp tối đa 100 lần hoặc đến khi |mean(xPhys) - volfrac| < 1e-6
    for _ in range(100):
        lmid = (l1 + l2) / 2

        # Quy tắc cập nhật OC (xem use_sqrt ở docstring)
        ratio = np.maximum(0.0, -dc / (dv * lmid + 1e-15))
        if use_sqrt:
            ratio = np.sqrt(ratio)
        xnew = np.maximum(
            X_MIN,
            np.maximum(
                x - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        x + move,
                        x * ratio,
                    ),
                ),
            ),
        )

        # Áp dụng bộ lọc mật độ
        if ft == 1:
            xPhys = xnew.copy()
        elif ft == 2:
            xPhys_flat = H @ xnew.flatten('F') / Hs
            xPhys = np.reshape(xPhys_flat, (nely, nelx), order='F')

        # Q được evaluate tại x cũ, không phải xnew - approximation chuẩn của
        # OC update (Sigmund 2001, Andreassen 2011), chấp nhận được với move
        # limit nhỏ (0.05-0.2).
        # FIX (xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục B1): khi có
        # projection, ràng buộc thể tích phải nhắm vào mean(x̂) (SAU projection,
        # trường thật sự dùng ở FE/chế tạo), không phải mean(x̃) (TRƯỚC
        # projection) - vì projection không bảo toàn thể tích tuyệt đối, dùng
        # x̃ để bisection gây lệch volfrac có hệ thống (đo được trong pilot).
        if projection == 'heaviside':
            x_hat = apply_heaviside_projection(xPhys, beta_proj, eta_proj)
            vol = np.mean(x_hat)
        else:
            vol = np.mean(xPhys)

        # MATLAB-style: mean(xPhys) > volfrac && Q(1,1) >= delta && Q(2,2) >= delta
        if has_stiffness_constraint:
            stiff_ok = (Q[0, 0] >= delta) and (Q[1, 1] >= delta)
        else:
            stiff_ok = True

        if vol > volfrac and stiff_ok:
            l1 = lmid
        else:
            l2 = lmid

        # Dừng sớm nếu Lagrange multiplier đã đạt độ chính xác cao
        if abs(vol - volfrac) < 1e-6 or (l2 - l1) < 1e-12:
            break

    return xnew, xPhys


def oc_update_dual(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    dg: np.ndarray,
    volfrac: float,
    delta: float,
    move: float,
    H,
    Hs,
    ft: int,
    Q: np.ndarray,
    use_sqrt: bool = False,
    projection: str | None = None,
    beta_proj: float = 8.0,
    eta_proj: float = 0.5,
    mu2_max: float = 1e9,
    max_outer: int = 100,
    max_inner: int = 60,
):
    """OC hai-multiplier (dual-constraint): volume VÀ stiffness cùng có hệ
    số Lagrange riêng, thay cho cổng nhị phân stiff_ok của oc_update()
    (stiffness_mode='gate') - xem docstring oc_update() cho phân tích cơ chế
    dao động của cách làm cũ.

    Công thức OC nhiều ràng buộc kinh điển (Bruyneel & Duysinx 2005; Bendsøe
    & Sigmund 2003 §5.3 - mở rộng tự nhiên của tiêu chí tối ưu 1 ràng buộc):

        ratio_e = max(0, -dc_e / max(mu1*dv_e - mu2*dg_e, eps))
        x_new_e = clip(x_e * ratio_e (hoặc sqrt(ratio_e) nếu use_sqrt), x_e-move, x_e+move, [X_MIN, 1])

    LƯU Ý DẤU: dg là độ nhạy THÔ dQ11/dx (hoặc dQ22/dx, tùy thành phần nào
    đang RÀNG BUỘC CHẶT HƠN - min(Q11,Q22)) theo x, ĐÃ qua projection-
    derivative + sensitivity-filter (giống dc) nhưng KHÔNG đổi dấu - caller
    (runner.py) chỉ cần truyền dQ[binding,binding] thẳng, không cần tự đảo
    dấu. dg thường DƯƠNG (thêm vật liệu thường tăng độ cứng), nên công thức
    TRỪ mu2*dg (không phải CỘNG như số hạng volume mu1*dv) - khác chiều với
    ràng buộc volume: volume là ràng buộc "dùng ít tài nguyên hơn khi
    multiplier tăng" (dv>0, CỘNG vào mẫu số làm ratio giảm khi mu1 tăng),
    còn stiffness là ràng buộc "cần NHIỀU vật liệu hơn khi vi phạm" (multiplier
    phải làm ratio TĂNG ở nơi dg lớn khi mu2 tăng) - nếu lỡ dùng dấu CỘNG ở
    đây, mu2 tăng sẽ đẩy ÍT vật liệu hơn đúng lúc cần nhiều hơn (ngược hoàn
    toàn ý đồ ràng buộc, đã bắt lỗi này khi viết unit test cho hàm này).

    Ràng buộc stiffness tại xnew được xấp xỉ TUYẾN TÍNH bậc 1 quanh x (Q
    không thể evaluate lại tại xnew mà không giải FE, giống approximation
    chuẩn của oc_update() cho Q ở chế độ 'gate' - nhưng ở đây bậc 1 thay vì
    bậc 0):

        g(xnew) ~= g(x) + sum(dg * (xnew - x)),  g = min(Q11, Q22)

    Vòng lặp NESTED: outer bisect mu1 (khớp volume, giống oc_update() gốc),
    inner bisect mu2 CHO MỖI mu1 (khớp xấp xỉ stiffness) - không có nghiệm
    dạng đóng khi 2 ràng buộc cùng active. Nếu ràng buộc KHÔNG vi phạm tại x
    hiện tại (complementary slackness: multiplier=0), bỏ qua inner loop
    (mu2=0), tương đương hành vi 'gate' khi thỏa mãn.

    So với 'gate': KHÔNG BAO GIỜ bỏ hoàn toàn ràng buộc thể tích trong 1 vòng
    lặp (mu1 luôn được bisect), và phân bổ vật liệu bổ sung cho stiffness
    THEO TRỌNG SỐ dg (không đồng loạt +move như 'gate' khi sụp về lmid->0).

    Returns:
        Bộ (xnew, xPhys) - cùng quy ước với oc_update().
    """
    nely, nelx = x.shape
    binding = 0 if Q[0, 0] <= Q[1, 1] else 1
    g_val = Q[binding, binding]
    slack0 = delta - g_val  # > 0 nghĩa là ĐANG VI PHẠM (cần thêm vật liệu)

    # FIX (bắt được qua pilot thật, xem PILOT_REPORT_dual_multiplier.md): mục
    # tiêu tuyến tính hóa `slack0` có thể VƯỢT XA những gì đạt được trong 1
    # vòng move-limit - max khả thi (mọi phần tử dg>0 nhảy hết lên x+move)
    # xấp xỉ sum(max(dg,0))*move. Nếu slack0 vượt mức này, inner bisection
    # đuổi theo 1 mục tiêu BẤT KHẢ THI, mu2 bị đẩy tới trần mu2_max, khiến
    # denom = mu1*dv - mu2*dg ÂM ở gần như MỌI phần tử có dg>0 (thường là đa
    # số, không phải thiểu số) - công thức suy biến thành "chỉ theo dấu dc",
    # XÓA SỔ hoàn toàn ảnh hưởng của mu1 (multiplier volume) khỏi mẫu số,
    # khiến outer bisection mất khả năng kiểm soát thể tích - đã đo được:
    # sụp cấu trúc 64% (vol->0.001) trên pilot N=100 đầu tiên (không cắt mục
    # tiêu). Cắt mục tiêu về mức khả thi (90% max, chừa biên an toàn) đảm bảo
    # tồn tại 1 mu2 HỮU HẠN thỏa mãn xấp xỉ, giữ mu1 luôn có ảnh hưởng thật.
    max_achievable = float(np.sum(np.maximum(dg, 0.0))) * move
    slack_target = min(slack0, 0.9 * max_achievable) if slack0 > 0.0 else 0.0

    def ratio_for(mu1, mu2):
        # Trừ (không cộng) mu2*dg - xem "LƯU Ý DẤU" ở docstring.
        denom = np.maximum(dv * mu1 - dg * mu2, 1e-6)
        r = np.maximum(0.0, -dc / denom)
        return np.sqrt(r) if use_sqrt else r

    def xnew_for(mu1, mu2):
        ratio = ratio_for(mu1, mu2)
        return np.clip(
            x * ratio,
            np.maximum(X_MIN, x - move),
            np.minimum(1.0, x + move),
        )

    def filtered(xnew_):
        if ft == 1:
            return xnew_.copy()
        flat = H @ xnew_.flatten('F') / Hs
        return np.reshape(flat, (nely, nelx), order='F')

    def vol_of(xPhys_):
        if projection == 'heaviside':
            return np.mean(apply_heaviside_projection(xPhys_, beta_proj, eta_proj))
        return np.mean(xPhys_)

    mu1_lo, mu1_hi = 0.0, 1e9
    xnew, xPhys = x.copy(), x.copy()
    for _ in range(max_outer):
        mu1 = (mu1_lo + mu1_hi) / 2

        if slack0 <= 0.0:
            # Ràng buộc không active tại x hiện tại - multiplier=0
            # (complementary slackness), tránh inner loop kéo cân bằng sai
            # hướng khi stiffness đã đủ dư.
            xnew = xnew_for(mu1, 0.0)
        else:
            mu2_lo, mu2_hi = 0.0, mu2_max
            for _ in range(max_inner):
                mu2 = (mu2_lo + mu2_hi) / 2
                xnew = xnew_for(mu1, mu2)
                # slack dự đoán tại xnew (xấp xỉ tuyến tính, xem docstring);
                # so với slack_target (ĐÃ CẮT về mức khả thi, không phải
                # slack0 thô - xem chú thích FIX ở trên) - s > 0 nghĩa là vẫn
                # chưa đạt mục tiêu (khả thi) -> cần multiplier LỚN hơn.
                s = slack_target - np.sum(dg * (xnew - x))
                if s > 0:
                    mu2_lo = mu2
                else:
                    mu2_hi = mu2
                if abs(s) < 1e-6 or (mu2_hi - mu2_lo) < 1e-12:
                    break

        xPhys = filtered(xnew)
        vol = vol_of(xPhys)
        if vol > volfrac:
            mu1_lo = mu1
        else:
            mu1_hi = mu1
        if abs(vol - volfrac) < 1e-6 or (mu1_hi - mu1_lo) < 1e-12:
            break

    return xnew, xPhys
