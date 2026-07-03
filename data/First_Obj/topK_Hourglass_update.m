function topK_Hourglass_update(nelx, nely, volfrac, penal, rmin, ft, nu, beta, ...
                               rotation_deg, void_size_frac, idx)
% SCAFFOLD BODY — identical across all 14 case files.

scale_factor = 8;
max_iter    = 80;
tolerance   = 0.05;
window_size = 20;

% PER-PATTERN 1: is_radial flag
is_radial = false;

% Folder name (spec §8)
ts = datestr(now,'yymmdd_HHMMSS');
folder_base = sprintf('res_%04d_V%.2f_P%d_R%.2f_N%.2f_B%.2f_T%d_S%.2f_%s', ...
    idx, volfrac, penal, rmin, nu, beta, rotation_deg, void_size_frac, ts);
folderName = folder_base;
mkdir(folderName);

% Material + FE setup (E0 = 193 MPa matches Zeng 304 SS GPa->MPa)
E0 = 193; Emin = 1e-9;
A11 = [12 3 -6 -3; 3 12 3 0; -6 3 12 -3; -3 0 -3 12];
A12 = [-6 -3 0 3; -3 -6 -3 -6; 0 -3 -6 3; 3 -6 3 -6];
B11 = [-4 3 -2 9; 3 -4 -9 4; -2 -9 -4 -3; 9 4 -3 -4];
B12 = [2 -3 4 -9; -3 2 9 -2; 4 9 2 3; -9 -2 3 2];
KE = 1/(1-nu^2)/24*([A11 A12; A12' A11]+nu*[B11 B12;B12' B11]);

% DOF + filter setup
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k = k+1;
                iH(k) = e1; jH(k) = e2; sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH,jH,sH); Hs = sum(H,2);

% PBC dofs
e0_mat = eye(3); ufixed = zeros(8,3); U = zeros(2*(nely+1)*(nelx+1),3);
alldofs = (1:2*(nely+1)*(nelx+1));
n1 = [nodenrs(end,[1,end]), nodenrs(1,[end,1])];
d1 = reshape([(2*n1-1); 2*n1],1,8);
n3 = [nodenrs(2:end-1,1)', nodenrs(end, 2:end-1)];
d3 = reshape([(2*n3-1); 2*n3],1,2*(nelx+nely-2));
n4 = [nodenrs(2:end-1,end)', nodenrs(1,2:end-1)];
d4 = reshape([(2*n4-1); 2*n4],1,2*(nelx+nely-2));
d2 = setdiff(alldofs,[d1,d3,d4]);
for j = 1:3
    ufixed(3:4,j) = [e0_mat(1,j),e0_mat(3,j)/2;e0_mat(3,j)/2,e0_mat(2,j)]*[nelx;0];
    ufixed(7:8,j) = [e0_mat(1,j),e0_mat(3,j)/2;e0_mat(3,j)/2,e0_mat(2,j)]*[0;nely];
    ufixed(5:6,j) = ufixed(3:4,j)+ufixed(7:8,j);
end
wfixed = [repmat(ufixed(3:4,:),nely-1,1); repmat(ufixed(7:8,:),nelx-1,1)];

% =========================================================================
% PER-PATTERN 2: SEED MASK — topK_Hourglass
% Hourglass void: bounding box 2*half_w × 2*half_h, tapers linearly to waist at center.
% At vsf=0.5 -> half_w=25, half_h=25 -> bbox 50×50 px (~50% cell each axis).
% waist_w = 0.1 * half_w keeps waist narrow at any vsf (cone-shaped pinch).
[I, J] = meshgrid(1:nelx, 1:nely);
center_x = nelx / 2; center_y = nely / 2;
half_w = void_size_frac * nelx / 2;
half_h = void_size_frac * nely / 2;
waist_w = 0.1 * half_w;

vertical_dist = abs(J - center_y);
taper_width = waist_w + (half_w - waist_w) .* (vertical_dist / half_h);
seed_mask = (abs(I - center_x) < taper_width) & (vertical_dist < half_h);
% END PER-PATTERN 2
% =========================================================================

% Apply PBC-correct rotation (helper from _helpers/)
seed_mask = pbc_rotate_mask(seed_mask, rotation_deg, is_radial);

% Project mask onto volfrac field
x = repmat(volfrac, nely, nelx);
x(seed_mask) = volfrac / 2;
xPhys = x; change = 1; loop = 0;

% Pre-allocate CSV buffer (18 cols)
csv_header = {'Iteration', 'File', 'nelx', 'nely', 'volfrac', 'penal', 'rmin', 'ft', 'nu', 'beta', ...
              'rotation_deg', 'void_size_frac', ...
              'Poisson_v12', 'Poisson_v21', 'is_auxetic', ...
              'Objective', 'MeanDensity', 'Objective_standardised'};
csv_data = cell(max_iter + 1, 18);

% Row 0: initial seed snapshot (NaN for Poisson + Obj + Obj_std, MeanDensity = mean(xPhys))
csv_data(1, :) = {0, mfilename, nelx, nely, volfrac, penal, rmin, ft, nu, beta, ...
                  rotation_deg, void_size_frac, ...
                  NaN, NaN, NaN, NaN, mean(xPhys(:)), NaN};
imwrite(imresize(1 - xPhys, scale_factor, 'nearest'), ...
        fullfile(folderName, sprintf('i_%05d.png', 0)));
row_idx = 1;

xi = 0.1; Vmax = volfrac; delta = xi * Vmax * E0;
prev_obj = inf; change_in_obj = inf; obj_changes = [];
converged_strict = false; converged_loose = false;
v12 = NaN; v21 = NaN; is_auxetic_flag = NaN; c = NaN;

while (change > 0.01) && (loop < max_iter)
    loop = loop + 1;

    % FEA solve
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    Kr = [K(d2,d2), K(d2,d3)+K(d2,d4); K(d3,d2)+K(d4,d2), K(d3,d3)+K(d4,d3)+K(d3,d4)+K(d4,d4)];
    U(d1,:) = ufixed;
    U([d2,d3],:) = Kr \ (-[K(d2,d1); K(d3,d1)+K(d4,d1)]*ufixed - [K(d2,d4); K(d3,d4)+K(d4,d4)]*wfixed);
    U(d4,:) = U(d3,:) + wfixed;

    qe = cell(3,3); Q = zeros(3,3); dQ = cell(3,3);
    for i = 1:3
        for j = 1:3
            U1 = U(:,i); U2 = U(:,j);
            qe{i,j} = reshape(sum((U1(edofMat)*KE).*U2(edofMat),2),nely,nelx)/(nelx*nely);
            Q(i,j) = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*qe{i,j}));
            dQ{i,j} = penal*(E0-Emin)*xPhys.^(penal-1).*qe{i,j};
        end
    end

    % =========================================================================
    % PER-PATTERN 3: OBJECTIVE — First_Obj (_stiff)
    c  = Q(1,2) - (beta^loop) * (Q(1,1) + Q(2,2));
    dc = dQ{1,2} - (beta^loop) * (dQ{1,1} + dQ{2,2});
    % END PER-PATTERN 3
    % =========================================================================

    dv = ones(nely,nelx);
    v12 = Q(2,1) / Q(1,1); v21 = Q(1,2) / Q(2,2);
    is_auxetic_flag = double((v12 < 0) && (v21 < 0));

    % Convergence tracking
    if prev_obj ~= inf
        change_in_obj = abs(c - prev_obj) / abs(prev_obj);
        obj_changes = [obj_changes, change_in_obj];
        if length(obj_changes) > window_size
            obj_changes(1) = [];
        end
        if length(obj_changes) == window_size && all(obj_changes < tolerance)
            converged_strict = true;
            break;
        end
    end
    prev_obj = c;

    % Snapshot row when loop == 1 or every 10
    is_save_iter = (loop == 1) || (mod(loop,10) == 0);
    if is_save_iter
        row_idx = row_idx + 1;
        csv_data = save_iteration(csv_data, row_idx, folderName, scale_factor, ...
            loop, mfilename, nelx, nely, volfrac, penal, rmin, ft, nu, beta, ...
            rotation_deg, void_size_frac, v12, v21, is_auxetic_flag, c, ...
            mean(xPhys(:)), Q(1,1), Q(2,2), xPhys);
    end

    % Filter sensitivities
    if ft == 1
        dc(:) = H*(x(:).*dc(:)) ./ Hs ./ max(1e-3, x(:));
    elseif ft == 2
        dc(:) = H*(dc(:) ./ Hs);
        dv(:) = H*(dv(:) ./ Hs);
    end

    % Lagrange bisection
    l1 = 0; l2 = 1e9; move = 0.1;
    while (l2 - l1 > 1e-9)
        lmid = 0.5 * (l2 + l1);
        xnew = max(0, max(x - move, min(1, min(x + move, x .* (-dc ./ dv / lmid)))));
        if ft == 1
            xPhys = xnew;
        elseif ft == 2
            xPhys(:) = (H * xnew(:)) ./ Hs;
        end
        if mean(xPhys(:)) > volfrac && Q(1,1) >= delta && Q(2,2) >= delta
            l1 = lmid;
        else
            l2 = lmid;
        end
    end

    change = max(abs(xnew(:) - x(:)));
    x = xnew; xPhys = xnew;
end

% Final snapshot if last loop is not already a snapshot
if ~((loop == 1) || (mod(loop,10) == 0))
    row_idx = row_idx + 1;
    csv_data = save_iteration(csv_data, row_idx, folderName, scale_factor, ...
        loop, mfilename, nelx, nely, volfrac, penal, rmin, ft, nu, beta, ...
        rotation_deg, void_size_frac, v12, v21, is_auxetic_flag, c, ...
        mean(xPhys(:)), Q(1,1), Q(2,2), xPhys);
end

% 2-tier loose convergence test (only if not strict)
if ~converged_strict && length(obj_changes) >= 10 && loop >= 50
    converged_loose = (std(obj_changes(end-9:end)) < 2*tolerance) && ...
                      (abs(mean(xPhys(:)) - volfrac) < 0.02);
end

% OUT_BASE from env var (Main_Batch sets it; smoke ad-hoc sets via setenv)
OUT_BASE = getenv('SIMP_OUT_BASE');
if isempty(OUT_BASE), OUT_BASE = pwd; end
geom_name = mfilename;

% Bucket + final CSV (helper)
classify_and_bucket(folderName, OUT_BASE, geom_name, ...
                    converged_strict, converged_loose, ...
                    csv_data(1:row_idx, :), csv_header, idx);
end
