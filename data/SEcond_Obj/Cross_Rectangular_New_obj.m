%% PERIODIC MATERIAL MICROSTRUCTURE DESIGN
function Cross_Rectangular_New_obj(nelx, nely, volfrac, penal, rmin, ft)
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
folderName = ['result_', datestr(now, 'yyyy-mm-dd_HHMMSS')];
mkdir(folderName);
disp(['Saving figures to: ', folderName]);
x = repmat(volfrac, nely, nelx);

% Parameters
horizontal_rect_width = floor(min(nelx, nely) / 4); % Width of horizontal rectangular voids
horizontal_rect_length = floor(min(nelx, nely) / 7); % Length of horizontal rectangular voids
vertical_rect_width = floor(min(nelx, nely) / 6); % Width of vertical rectangular voids (rotated 90 degrees)
vertical_rect_length = floor(min(nelx, nely) / 4); % Length of vertical rectangular voids
center_x = nelx / 2; % Center of the design domain (x-direction)
center_y = nely / 2; % Center of the design domain (y-direction)

% Create horizontal rectangular voids in the middle of the edges
horizontal_positions = [
    center_x, 1; % Top edge
    center_x, nely; % Bottom edge
    1, center_y; % Left edge
    nelx, center_y % Right edge
];

for pos = 1:size(horizontal_positions, 1)
    cx = horizontal_positions(pos, 1);
    cy = horizontal_positions(pos, 2);

    for i = 1:nelx
        for j = 1:nely
            if (pos == 1 || pos == 2) && ...
               abs(j - cy) <= horizontal_rect_width / 2 && ...
               abs(i - cx) <= horizontal_rect_length
                x(j, i) = volfrac / 2; % Horizontal rectangles
            elseif (pos == 3 || pos == 4) && ...
                   abs(i - cx) <= horizontal_rect_width / 2 && ...
                   abs(j - cy) <= horizontal_rect_length
                x(j, i) = volfrac / 2; % Horizontal rectangles
            end
        end
    end
end

% Create vertical rectangles on top of the horizontal voids
vertical_positions = horizontal_positions; % Same positions as horizontal rectangles

for pos = 1:size(vertical_positions, 1)
    cx = vertical_positions(pos, 1);
    cy = vertical_positions(pos, 2);

    for i = 1:nelx
        for j = 1:nely
            if (pos == 1 || pos == 2) && ...
               abs(i - cx) <= vertical_rect_width / 2 && ...
               abs(j - cy) <= vertical_rect_length
                x(j, i) = volfrac / 2; % Vertical rectangles on top of horizontal ones
            elseif (pos == 3 || pos == 4) && ...
                   abs(j - cy) <= vertical_rect_width / 2 && ...
                   abs(i - cx) <= vertical_rect_length
                x(j, i) = volfrac / 2; % Vertical rectangles on top of horizontal ones
            end
        end
    end
end
xPhys = x;

change = 1;
loop = 0;

%% Initialize arrays to store iteration data
iterations = [];
volume_fractions = [];
poisson_ratios_v12 = [];
poisson_ratios_v21 = [];

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

    %% Calculate delta and apply constraints
    delta = 0.1 * volfrac * E0;
    
    c = Q(1,2);  % Objective function: Maximize shear stiffness
    
    %% Constraint for Q(1,1) and Q(2,2)
    if loop < 20 % Control for the first 20 iterations
        scale_factor = 1 - 0.02 * loop; % Scale down influence over time
        c = c - scale_factor * (Q(1,1) + Q(2,2)); % Reduce influence gradually
    end

    % Apply constraints for horizontal (E_1111^H) and vertical (E_2222^H) stiffness
    penalty = 1e2;  % Penalty factor for constraint violations
    if Q(1,1) < delta
        c = c + penalty * (delta - Q(1,1))^2;  % Penalize if E_1111^H < delta
        fprintf('Iteration: %d - Horizontal stiffness penalty applied!\n', loop);
    end
    if Q(2,2) < delta
        c = c + penalty * (delta - Q(2,2))^2;  % Penalize if E_2222^H < delta
    end

    dc = dQ{1,2};  % Sensitivity analysis
    if loop < 20
        dc = dc - scale_factor * (dQ{1,1} + dQ{2,2}); % Adjust derivative accordingly
    end

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