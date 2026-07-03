clc
clear all
nelx=100; nely=100; volfrac=0.4; penal=3; rmin=3; ft=2;
topK_Hourglass_New_obj(nelx, nely, volfrac, penal, rmin, ft)
%% nelx, nely: number of elements along x, y
%% volfrac: prescribed volume fraction (maximum alowable Vol fraction)
%% penal: strength of penalty
%% rmin: the radius of the filter
%% ft: filtering choices