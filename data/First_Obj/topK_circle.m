function topK_circle(nelx,nely,volfrac,penal,rmin,ft,idx)
% -------- SETTINGS -------------------------------------------------------
E0 = 199; Emin = 1e-8;    % tăng Emin để tránh singular
nu = 0.3;
scale_factor = 8;
max_iter = 100; tol_change = 0.05; win_size = 20;

ts = datestr(now,'yymmdd_HHMMSS');
base = sprintf('res_%04d_V%.2f_P%d_R%.2f_%s',idx,volfrac,penal,rmin,ts);
folder = base; mkdir(folder);

% -------- KE & FE PREP ---------------------------------------------------
A11=[12 3 -6 -3;3 12 3 0;-6 3 12 -3;-3 0 -3 12];
A12=[-6 -3 0 3;-3 -6 -3 -6;0 -3 -6 3;3 -6 3 -6];
B11=[-4 3 -2 9;3 -4 -9 4;-2 -9 -4 -3;9 4 -3 -4];
B12=[2 -3 4 -9;-3 2 9 -2;4 9 2 3;-9 -2 3 2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);

nodenrs=reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofV = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofM = repmat(edofV,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofM,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofM,ones(1,8))',64*nelx*nely,1);

% -------- FILTER ---------------------------------------------------------
[iGrid,jGrid] = meshgrid(1:nelx,1:nely);
H=sparse([],[],[],nelx*nely,nelx*nely);
for dx=-(ceil(rmin)-1):(ceil(rmin)-1)
  for dy=-(ceil(rmin)-1):(ceil(rmin)-1)
     rr=sqrt(dx^2+dy^2);
     if rr>=rmin || rr==0, continue; end
     ish=max(min(iGrid+dx,nelx),1); jsh=max(min(jGrid+dy,nely),1);
     id1=sub2ind([nely,nelx],jGrid(:),iGrid(:));
     id2=sub2ind([nely,nelx],jsh(:),ish(:));
     H=H+sparse(id1,id2,(rmin-rr),nelx*nely,nelx*nely);
  end
end
Hs=sum(H,2);

% -------- PBC ------------------------------------------------------------
e0=eye(3); uf=zeros(8,3); U=zeros(2*(nely+1)*(nelx+1),3);
alld=1:2*(nely+1)*(nelx+1);
n1=[nodenrs(end,[1,end]),nodenrs(1,[end,1])];
d1=reshape([(2*n1-1);2*n1],1,8);
n3=[nodenrs(2:end-1,1)',nodenrs(end,2:end-1)];
d3=reshape([(2*n3-1);2*n3],1,2*(nelx+nely-2));
n4=[nodenrs(2:end-1,end)',nodenrs(1,2:end-1)];
d4=reshape([(2*n4-1);2*n4],1,2*(nelx+nely-2));
d2=setdiff(alld,[d1,d3,d4]);
for j=1:3
  uf(3:4,j)=[e0(1,j),e0(3,j)/2;e0(3,j)/2,e0(2,j)]*[nelx;0];
  uf(7:8,j)=[e0(1,j),e0(3,j)/2;e0(3,j)/2,e0(2,j)]*[0;nely];
  uf(5:6,j)=uf(3:4,j)+uf(7:8,j);
end
wfix=[repmat(uf(3:4,:),nely-1,1);repmat(uf(7:8,:),nelx-1,1)];

% -------- INITIAL CIRCLE -------------------------------------------------
x = volfrac*ones(nely,nelx);
cx=nelx/2; cy=nely/2; rad=min(nelx,nely)/6;
[I,J]=meshgrid(1:nelx,1:nely);
x( (I-cx).^2 + (J-cy).^2 <= rad^2 ) = volfrac/2;
xPhys=x;

% -------- CSV & IMG init -------------------------------------------------
csvH={'Iteration','File','nelx','nely','volfrac','penal','rmin','ft',...
      'Poisson_v12','Poisson_v21','Objective','MeanDensity'};
csvD={0,mfilename,nelx,nely,volfrac,penal,rmin,ft,NaN,NaN,NaN,mean(xPhys(:))};
imwrite(imresize(1-xPhys,scale_factor,'nearest'), fullfile(folder,'i_00000.png'));

% -------- ITERATION LOOP -------------------------------------------------
prev=inf; hist=[]; loop=0; change=1;
while change>0.01 && loop<max_iter
  loop=loop+1;
  sK=reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
  K=sparse(iK,jK,sK); K=(K+K')/2 + speye(size(iK,1))*Emin; % epsilon on diag
  Kr=[K(d2,d2),K(d2,d3)+K(d2,d4);K(d3,d2)+K(d4,d2), ...
      K(d3,d3)+K(d4,d3)+K(d3,d4)+K(d4,d4)];
  U(d1,:)=uf;
  U([d2,d3],:)=Kr\(-[K(d2,d1);K(d3,d1)+K(d4,d1)]*uf ...
                   -[K(d2,d4);K(d3,d4)+K(d4,d4)]*wfix);
  U(d4,:)=U(d3,:)+wfix;

  % homogenised Q
  Q=zeros(3,3); dQ=cell(3,3);
  for a = 1:3
    for b = 1:3
        Ue1 = U(edofMat, a);          % (nelx*nely)×8 matrix
        Ue2 = U(edofMat, b);
        qe  = reshape(sum((Ue1*KE).*Ue2,2), nely, nelx)/(nelx*nely);
        Q(a,b)=sum(sum((Emin+xPhys.^penal*(E0-Emin)).*qe));
        dQ{a,b}=penal*(E0-Emin)*xPhys.^(penal-1).*qe;
    end
  end
  c = Q(1,2) - (0.8^loop)*(Q(1,1)+Q(2,2));
  dc= dQ{1,2} - (0.8^loop)*(dQ{1,1}+dQ{2,2});

  % rolling stop
  hist=[hist,abs(c-prev)/max(abs(prev),eps)]; if numel(hist)>win_size,hist(1)=[];end
  if numel(hist)==win_size && all(hist<tol_change), break; end
  prev=c;

  % Poisson
  v12=Q(2,1)/Q(1,1); v21=Q(1,2)/Q(2,2);

  % save key iter
  if loop==1 || mod(loop,10)==0
    csvD=[csvD; {loop,mfilename,nelx,nely,volfrac,penal,rmin,ft,...
                 v12,v21,c,mean(xPhys(:))}];
    imwrite(imresize(1-xPhys,scale_factor,'nearest'), ...
            fullfile(folder,sprintf('i_%05d.png',loop)));
  end

  % filter
  dv=ones(nely,nelx);
  dc(:)=H*( (ft==1).*x(:).*dc(:) + (ft==2).*dc(:) )./Hs./max(1e-3,x(:));
  if ft==2, dv(:)=H*(dv(:)./Hs); end

  % OC update
  l1=0;l2=1e9;move=0.1;
  while l2-l1>1e-9
    lm=0.5*(l1+l2);
    xnew=max(0,max(x-move,min(1,min(x+move,x.*(-dc./dv/lm)))));
    if ft==2, xPhys=reshape((H*xnew(:))./Hs,nely,nelx);end
    if mean(xPhys(:))>volfrac,l1=lm;else,l2=lm;end
  end
  change=max(abs(xnew(:)-x(:))); x=xnew; if ft==1, xPhys=xnew; end
end

% save cuối
if mod(loop,10)~=0
  csvD=[csvD; {loop,mfilename,nelx,nely,volfrac,penal,rmin,ft,...
               v12,v21,c,mean(xPhys(:))}];
  imwrite(imresize(1-xPhys,scale_factor,'nearest'), ...
          fullfile(folder,sprintf('i_%05d.png',loop)));
end

% classify
noConv=(loop==max_iter); noAux=(v12>0)||(v21>0);
suffix=''; if noConv, suffix=[suffix '_no_converge']; end
if noAux, suffix=[suffix '_noauxetic']; end
if ~isempty(suffix), new=[base suffix]; movefile(folder,new); folder=new; end
dest=ternary(~noConv && ~noAux,'result_OK','result_NG');
if ~exist(dest,'dir'), mkdir(dest); end
movefile(folder,fullfile(dest,folder));
writecell([csvH;csvD], fullfile(dest,folder,'sum.csv'));
fprintf('>> Case %d done | obj %.3f | v12 %.3f | v21 %.3f | iter %d\n',...
        idx,c,v12,v21,loop);
end

function out=ternary(c,a,b); if c,out=a; else, out=b; end, end
