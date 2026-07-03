%% PERIODIC MATERIAL MICROSTRUCTURE DESIGN
function small_square_cross_arrange(nelx, nely, volfrac, penal, rmin, ft)
%% nelx, nely: number of elements along x, y
%% MATERIAL PROPERTIES
E0 = 199;
Emin = 1e-9;
nu = 0.3;

%% PREPARE FINITE ELEMENT ANALYSIS
A11 = [12 3 -6 -3; 3 12 3 0; -6 3 12 -3; -3 0 -3 12];
A12 = [-6 -3 0 3; -3 -6 -3 -6; 0 -3 -6 3; 3 -6 3 -6];
B11 = [-4 3 -2 9; 3 -4 -9 4; -2 -9 -4 -3; 9 4 -3 -4];
B12 = [2 -3 4 -9; -3 2 9 -2; 4 9 2 3; -9 -2 3 2];
KE = 1/(1-nu^2)/24*([A11 A12; A12' A11]+nu*[B11 B12;B12' B11]);

%% node numbering and element degrees of freedom
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

%% PREPARE FILTER
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
                iH(k) = e1;
                jH(k) = e2;
                sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH,jH,sH); 
Hs = sum(H,2);

%% PERIODIC BOUNDARY CONDITIONS
e0 = eye(3);
ufixed = zeros(8,3);
U = zeros(2*(nely+1)*(nelx+1),3);
alldofs = (1:2*(nely+1)*(nelx+1));
n1 = [nodenrs(end,[1,end]), nodenrs(1,[end,1])];
d1 = reshape([(2*n1-1); 2*n1],1,8);
n3 = [nodenrs(2:end-1,1)', nodenrs(end, 2:end-1)];
d3 = reshape([(2*n3-1); 2*n3],1,2*(nelx+nely-2)); 
n4 = [nodenrs(2:end-1,end)', nodenrs(1,2:end-1)];
d4 = reshape([(2*n4-1); 2*n4],1,2*(nelx+nely-2)); 
d2 = setdiff(alldofs,[d1,d3,d4]);

for j = 1:3
    ufixed(3:4,j) = [e0(1,j),e0(3,j)/2;e0(3,j)/2,e0(2,j)]*[nelx;0];
    ufixed(7:8,j) = [e0(1,j),e0(3,j)/2;e0(3,j)/2,e0(2,j)]*[0;nely];
    ufixed(5:6,j) = ufixed(3:4,j)+ufixed(7:8,j);
end
wfixed = [repmat(ufixed(3:4,:),nely-1,1); repmat(ufixed(7:8,:),nelx-1,1)];

%% INITIALIZE ITERATION
x = repmat(volfrac, nely, nelx);

% Parameters
outer_square_size = floor(min(nelx, nely) / 10); % Size of the outer corner squares
inner_square_size = floor(min(nelx, nely) / 14); % Size of the inner squares near the corners
beam_width = floor(min(nelx, nely) / 40); % Width of the thin rectangular beams
beam_length = floor(min(nelx, nely) / 3); % Length of the thin rectangular beams
center_x = nelx / 2; % Center of the design domain (x-direction)
center_y = nely / 2; % Center of the design domain (y-direction)

% Create outer corner squares
corner_positions = [
    floor(center_x / 5), floor(center_y / 5); % Top-left (moved closer to the center)
    floor(9 * center_x / 5), floor(center_y / 5); % Top-right (moved closer to the center)
    floor(center_x / 5), floor(9 * center_y / 5); % Bottom-left (moved closer to the center)
    floor(9 * center_x / 5), floor(9 * center_y / 5) % Bottom-right (moved closer to the center)
];

for pos = 1:size(corner_positions, 1)
    start_x = max(1, corner_positions(pos, 1) - outer_square_size / 2);
    end_x = min(nelx, corner_positions(pos, 1) + outer_square_size / 2);
    start_y = max(1, corner_positions(pos, 2) - outer_square_size / 2);
    end_y = min(nely, corner_positions(pos, 2) + outer_square_size / 2);

    x(start_y:end_y, start_x:end_x) = volfrac / 2;
end

% Create smaller inner squares near the corners
inner_square_positions = [
     floor(center_x / 4) + floor(outer_square_size * 1.5), floor(center_y / 4) + floor(outer_square_size * 1.5); % Top-left inner
    floor(7 * center_x / 4) - floor(outer_square_size * 1.5), floor(center_y / 4) + floor(outer_square_size * 1.5); % Top-right inner
    floor(center_x / 4) + floor(outer_square_size * 1.5), floor(7 * center_y / 4) - floor(outer_square_size * 1.5); % Bottom-left inner
    floor(7 * center_x / 4) - floor(outer_square_size * 1.5), floor(7 * center_y / 4) - floor(outer_square_size * 1.5) % Bottom-right inner
];

for pos = 1:size(inner_square_positions, 1)
    start_x = max(1, inner_square_positions(pos, 1) - inner_square_size / 2);
    end_x = min(nelx, inner_square_positions(pos, 1) + inner_square_size / 2);
    start_y = max(1, inner_square_positions(pos, 2) - inner_square_size / 2);
    end_y = min(nely, inner_square_positions(pos, 2) + inner_square_size / 2);

    x(start_y:end_y, start_x:end_x) = volfrac / 2;
end

% Create thin rectangular beams from the edges
% Top beam (from top edge to center)
x(1:beam_length, center_x - beam_width/2:center_x + beam_width/2) = volfrac / 2;

% Bottom beam (from bottom edge to center)
x(nely - beam_length:nely, center_x - beam_width/2:center_x + beam_width/2) = volfrac / 2;

% Left beam (from left edge to center)
x(center_y - beam_width/2:center_y + beam_width/2, 1:beam_length) = volfrac / 2;

% Right beam (from right edge to center)
x(center_y - beam_width/2:center_y + beam_width/2, nelx - beam_length:nelx) = volfrac / 2;

xPhys = x;

% Save results with the design name
folderName = ['result_', datestr(now, 'yyyy-mm-dd_HHMMSS')];
mkdir(folderName);
disp(['Saving figures to: ', folderName]);

change = 1;
loop = 0;

%% Initialize arrays to store iteration data
iterations = [];
volume_fractions = [];
poisson_ratios_v12 = [];
poisson_ratios_v21 = [];

%% Define Supplementary Constraint Parameters
xi = 0.02;  % Constraint parameter, can be adjusted as needed
Vmax = volfrac;  % Maximum volume fraction
delta = xi * Vmax * E0;  % Supplementary restriction constraint

%% Plot and Save Initial Design as Iteration 0
figureHandle = figure('visible','off');
colormap(gray);
imagesc(1 - xPhys);
caxis([0 1]);
axis equal; axis off;
fileName = sprintf('%s/iteration_%05d.png', folderName, 0);
saveas(figureHandle, fileName);
close(figureHandle);
fprintf(' It.:%5i Initial Design\n', 0);

%% Initialize variables for rolling average
tolerance = 0.05;  % 5% tolerance for stopping criterion
window_size = 20;  % Number of iterations to consider for rolling average
prev_obj = inf;   % Initialize the previous objective function value
change_in_obj = inf;  % Initialize the change in objective function value
obj_changes = [];  % List to store recent objective function changes

%% START ITERATION
while (change > 0.01)
    loop = loop+1;
    %% FE-ANALYSIS
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    Kr = [K(d2,d2), K(d2,d3)+K(d2,d4); K(d3,d2)+K(d4,d2), K(d3,d3)+K(d4,d3)+K(d3,d4)+K(d4,d4)];
    U(d1,:) = ufixed;
    U([d2,d3],:) = Kr\(-[K(d2,d1); K(d3,d1)+K(d4,d1)]*ufixed-[K(d2,d4); K(d3,d4)+K(d4,d4)]*wfixed);
    U(d4,:) = U(d3,:)+wfixed;
    
    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    for i = 1:3
        for j = 1:3
            U1 = U(:,i); U2 = U(:,j);
            qe{i,j} = reshape(sum((U1(edofMat)*KE).*U2(edofMat),2),nely,nelx)/(nelx*nely);
            Q(i,j) = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*qe{i,j}));
            dQ{i,j} = penal*(E0-Emin)*xPhys.^(penal-1).*qe{i,j};
        end
    end

    % Modified Objective Function
    c = Q(1,2) - (0.8^loop) * (Q(1,1) + Q(2,2));   % Only the term for E_1122^H
    dc = dQ{1,2} - (0.8^loop) * (dQ{1,1} + dQ{2,2}); % Sensitivity corresponds only to E_1122^H
    % 
    % % Additional Constraints for E_1111^H and E_2222^H
    % if Q(1,1) > delta  % E_1111^H must be less than or equal to delta
    %     dc = dc + (Q(1,1) - delta);  % Penalize if E_1111^H exceeds delta
    % end
    % if Q(2,2) > delta  % E_2222^H must be less than or equal to delta
    %     dc = dc + (Q(2,2) - delta);  % Penalize if E_2222^H exceeds delta
    % end

    % Calculate objective function change
    if prev_obj ~= inf
        change_in_obj = abs(c - prev_obj) / abs(prev_obj);  % Relative change in objective
         % Add the change to the list
        obj_changes = [obj_changes, change_in_obj];
         if length(obj_changes) > window_size
            obj_changes(1) = [];  % Remove the oldest entry if exceeding window size
         end
         % Check if all changes in the window are below the tolerance
        if length(obj_changes) == window_size && all(obj_changes < tolerance)
            fprintf('Stopping optimization: Objective function change below 5%% for %d consecutive iterations.\n', window_size);
            break;
        end
    end
    
    % Print the current change and the window length for debugging
    fprintf('Iteration: %d, Obj Change: %f, Window Length: %d\n', loop, change_in_obj, length(obj_changes));
    % Update the previous objective function value
    prev_obj = c;

    dv = ones(nely,nelx);

       %% Calculate Poisson's ratios
    v12 = Q(2,1) / Q(1,1);  % Poisson's ratio nu_12
    v21 = Q(1,2) / Q(2,2);  % Poisson's ratio nu_21
    
    % Save data to CSV named after .m file
    csvFileName = fullfile(folderName, [mfilename, '_iteration_data.csv']);
    header = {'Iteration', 'File', 'nelx', 'nely', 'volfrac', 'penal', 'rmin', ...
              'ft', 'Poisson_v12', 'Poisson_v21', 'Objective', 'MeanDensity'};
    row = {loop, mfilename, nelx, nely, volfrac, penal, rmin, ft, ...
           v12, v21, c, mean(xPhys(:))};
    if loop == 1 && ~isfile(csvFileName)
        writecell([header; row], csvFileName);
    else
        writecell(row, csvFileName, 'WriteMode', 'append');
    end
    
      %% Append data for current iteration
    iterations = [iterations; loop];
    volume_fractions = [volume_fractions; mean(xPhys(:))];
    poisson_ratios_v12 = [poisson_ratios_v12; v12];
    poisson_ratios_v21 = [poisson_ratios_v21; v21];

    %% Print Poisson's ratios for each iteration
    fprintf('Iteration: %d, Poisson ratio v12: %f, v21: %f\n', loop, v12, v21);

    %% FILTERING/MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end

    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1 = 0; l2 = 1e9; move = 0.1;
    while (l2-l1 > 1e-9)
        lmid = 0.5*(l2+l1);
        xnew = max(0, max(x-move, min(1, min(x+move, x.*(-dc./dv/lmid)))));
        if ft == 1
            xPhys = xnew;
        elseif ft == 2
            xPhys(:) = (H*xnew(:))./Hs;
        end
        % Apply volume and stiffness constraints
        if mean(xPhys(:)) > volfrac && Q(1,1) >= delta && Q(2,2) >= delta
            l1 = lmid;
        else
            l2 = lmid;
        end
    end

    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    xPhys = xnew;

    %% Plotting and Saving the Figure
    figureHandle = figure('visible', 'off');
    colormap(gray);
    imagesc(1 - xPhys);
    caxis([0 1]);
    axis equal; axis off;

    %% SAVE THE FIGURE TO A FILE
    fileName = sprintf('%s/iteration_%05d.png', folderName, loop);
    saveas(figureHandle, fileName);
    close(figureHandle);

    %% PRINT RESULTS TO THE CONSOLE
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.: %7.3f\n', loop, c, mean(xPhys(:)), change);
end