"""
Phase 5 - manufacturability.py
============================================================
Roadmap 6.2 + 6.3: 2 kiểm tra hình học trên density field ĐÃ binarize, độc
lập với FE/surrogate - không đánh giá Poisson ratio, chỉ đánh giá "cấu trúc
này có sản xuất/lắp ráp được thành lattice tuần hoàn hay không".

6.2 - connectivity / kích thước feature tối thiểu:
  Một density field có thể cho ra (v12, v21) đúng target qua FE thật (PBC
  chỉ ràng buộc trường chuyển vị lúc giải, không ràng buộc vật liệu phải
  liền khối) nhưng vẫn KHÔNG in/đúc được nếu vật liệu rời rạc thành nhiều
  mảnh không chạm nhau, hoặc có nét mảnh hơn độ phân giải gia công.

6.3 - periodicity/tiling:
  Đây là kiểm tra HÌNH HỌC khác với PBC toán học trong
  simp/core/pbc.py/verify_fe.py (PBC chỉ giả định trường chuyển vị tuần
  hoàn khi GIẢI FE trên 1 ô, không đảm bảo ảnh sinh ra ghép liền mạch khi
  in thành lattice thật). Ô đơn vị được lát bằng tịnh tiến (translation),
  nên khi ghép ô N tại cạnh phải với ô N+1 (bản sao y hệt) tại cạnh trái,
  cột phải (col=nelx-1) của ô N nằm sát cột trái (col=0) của ô N+1 - vốn
  cũng là cột trái của chính ô N. Vậy nếu occupancy cột phải và cột trái
  (tương ứng hàng trên/dưới) không khớp, sẽ có bước nhảy rắn/rỗng đột ngột
  ngay tại mép ghép - vẫn hợp lệ về mặt homogenization (chỉ là nơi 1 thanh
  kết thúc) nhưng đáng gắn cờ để người thiết kế xem lại trước khi sản xuất.
"""
import numpy as np
from scipy import ndimage


def check_connectivity(img_bin: np.ndarray, min_feature_px: int = 2):
    """img_bin: mảng nhị phân (0/1 hoặc bool), 1 = vật liệu rắn.

    - n_components: số mảnh vật liệu rời rạc (không kể pha rỗng), dùng
      connectivity 8-nối (kể cả chéo) vì đó là điều kiện "chạm nhau" đúng
      cho lưới vuông.
    - is_connected: True nếu chỉ có <=1 mảnh (0 mảnh = cấu trúc toàn rỗng,
      coi là suy biến, không tính "liên thông" nhưng vẫn is_connected=True
      vì không có mảnh rời).
    - min_feature_ok: sau khi erosion bằng min_feature_px // 2 lần (bán
      kính cấu trúc tương ứng độ rộng nét tối thiểu), nếu 1 mảnh biến mất
      hoàn toàn (không còn pixel nào) nghĩa là nét đó mảnh hơn ngưỡng gia
      công min_feature_px - đánh dấu False.
    """
    solid = np.asarray(img_bin) > 0.5
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labels, n_components = ndimage.label(solid, structure=structure)

    is_connected = n_components <= 1

    min_feature_ok = True
    thin_component_ids = []
    if n_components > 0 and min_feature_px > 1:
        erosion_iters = min_feature_px // 2
        eroded = ndimage.binary_erosion(solid, iterations=erosion_iters)
        for comp_id in range(1, n_components + 1):
            comp_mask = labels == comp_id
            if not np.any(eroded & comp_mask):
                # cả mảnh này biến mất sau khi erosion -> mảnh này mảnh
                # hơn ngưỡng min_feature_px, dù có thể vẫn còn "tồn tại"
                # trước erosion.
                min_feature_ok = False
                thin_component_ids.append(comp_id)

    return {
        "n_components": int(n_components),
        "is_connected": bool(is_connected),
        "min_feature_ok": bool(min_feature_ok),
        "n_thin_components": len(thin_component_ids),
        "manufacturable": bool(is_connected and min_feature_ok),
    }


def check_periodicity(img_bin: np.ndarray, tol: float = 0.1):
    """So sánh occupancy cột trái/phải và hàng trên/dưới - xem docstring
    đầu file. tol: tỉ lệ pixel-biên-không-khớp tối đa để vẫn coi là
    "tương thích tuần hoàn" (0.0 = yêu cầu khớp tuyệt đối từng pixel)."""
    solid = np.asarray(img_bin) > 0.5
    left, right = solid[:, 0], solid[:, -1]
    top, bottom = solid[0, :], solid[-1, :]

    mismatch_lr = float((left != right).mean())
    mismatch_tb = float((top != bottom).mean())

    periodic_ok = mismatch_lr <= tol and mismatch_tb <= tol

    return {
        "edge_mismatch_lr": mismatch_lr,
        "edge_mismatch_tb": mismatch_tb,
        "periodic_ok": bool(periodic_ok),
    }


def check_manufacturability(img_bin: np.ndarray, min_feature_px: int = 2,
                             periodicity_tol: float = 0.1):
    """Gộp cả 2 kiểm tra - dùng trực tiếp trong best_of_n_eval.py."""
    conn = check_connectivity(img_bin, min_feature_px=min_feature_px)
    period = check_periodicity(img_bin, tol=periodicity_tol)
    return {
        **conn,
        **period,
        "passes_all": bool(conn["manufacturable"] and period["periodic_ok"]),
    }


def force_periodic(img: np.ndarray) -> np.ndarray:
    """Ép img[:, -1]=img[:, 0] và img[0, :]=img[-1, :] (trung bình rồi gán
    lại cả 2 cạnh) - đảm bảo periodic_ok=True TUYỆT ĐỐI (edge_mismatch=0)
    theo đúng định nghĩa periodicity ở check_periodicity() phía trên: ô kề
    bên khi lát là BẢN SAO Y HỆT ô này (tịnh tiến), nên cạnh phải của ô
    PHẢI bằng cạnh trái của CHÍNH NÓ - đây không phải 1 ràng buộc cần model
    generate() học, mà là hệ quả toán học trực tiếp của phép lát, ép được
    bằng đúng 1 phép gán, không cần train lại.

    Phát hiện thực nghiệm (nhánh research/auxetic-breakthrough, xem
    EXPERIMENT_LOG.md mục "Phase 6"): áp trực tiếp hàm này lên ảnh cVAE
    sinh ra (checkpoint cvae_gamma20.pt, KHÔNG train lại) đưa passes_all từ
    1,7% lên 19,5% (đo trên 600 mẫu, 6 condition x 100 mẫu, seed=123) -
    cùng bậc cải thiện đo được trên kiến trúc diffusion thử nghiệm (1,8%->
    35,2%). Chi phí sai số property: mean |Δv12| (FE thật, 10 mẫu) = 0,0193
    - chỉ ghi đè ~3% số pixel (1 hàng + 1 cột trong lưới 64x64) nên tác
    động lên tensor độ cứng đồng nhất hóa toàn cục thường nhỏ, nhưng có thể
    lớn hơn ở mẫu có cơ cấu bản lề/re-entrant chạm đúng biên ô - nên đọc
    kèm CI/kiểm tra từng mẫu khi cần độ chính xác cao, không chỉ trung bình.
    """
    img = img.copy()
    avg_col = (img[:, 0] + img[:, -1]) / 2
    img[:, 0] = avg_col
    img[:, -1] = avg_col
    avg_row = (img[0, :] + img[-1, :]) / 2
    img[0, :] = avg_row
    img[-1, :] = avg_row
    return img
