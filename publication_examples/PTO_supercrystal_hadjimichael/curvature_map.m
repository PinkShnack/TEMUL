%% Curvature map in SrRuO3 layers - Marios Hadjimichael
% "Metal-ferroelectric supercrystals with periodically curved metallic
% layers"
% Reads the positions of Ru atoms from the Atomap output
% Fits a sinusoidal function to each atomic plane
% Calculates the curvature - second derivative of the sinusoidal
% function
%% clear all current variables and open the HDF5 file with atomic positions
clear;
% HDF5 file in the current folder
path = [pwd, '\'];
h5_filename = [path, 'Atom_Lattice.hdf5'];
info = h5info(h5_filename);

%% import the data
% original image
tem_image = h5read (h5_filename,'/original_image_data');
tem_image = tem_image';

% import the atomic positions
sublattice_01_atom_positions = h5read (h5_filename,'/1_sublattice/atom_positions');
sublattice_01_modified_image = h5read (h5_filename,'/1_sublattice/modified_image_data');
sublattice_01_modified_image = sublattice_01_modified_image';

%% define the image size to accurately map the scale 
% horizontal size of image in nanometres
image_size_x_nm = 29.278; 
image_size_x_m = image_size_x_nm*1e-9;
% find number of pixels in the horizontal direction
pixel_number_x = size(tem_image,2);
% convert pixels to nanometres and metres
nanometres_per_pixel = image_size_x_nm/pixel_number_x;

%% convert the x-y scale in the TEM image into nanometres
% x scale is 1994 pixels
% x coordinate defined as:
tem_image_x = 0:nanometres_per_pixel:(size(tem_image,2)-1)*nanometres_per_pixel;
tem_image_x = tem_image_x';

% y scale is 206 pixels
% y coordinate defined as:
tem_image_y = 0:nanometres_per_pixel:(size(tem_image,1)-1)*nanometres_per_pixel;
tem_image_y = tem_image_y';
%% redefine atomic positions in nanometres
sublattice_01_atom_positions = sublattice_01_atom_positions*nanometres_per_pixel;

%% split atomic positions plane by plane
% all x-positions of the atoms:
atom_positions_x = sublattice_01_atom_positions(1,:);
% all y-positions of the atoms:
atom_positions_y = sublattice_01_atom_positions(2,:);

% create a new x-vector from min(x) to max(x) so we can interpolate the x
% positions
% this interpolation will not be necessary in this case since we fit with a
% sine wave
min_x = min(atom_positions_x);
max_x = max(atom_positions_x);
step_size = 0.025;
%create new structure for the planes
planes = struct('X',{});
% load('planes.mat','planes');
% each item in the structure is a separate plane 
for i=1:6
   % x-positions of each plane - each plane has 81 atoms 
   planes(i).plane_x =  atom_positions_x((i-1)*81+1:i*81);
   planes(i).plane_x = planes(i).plane_x';
   % y-positions of each plane
   planes(i).plane_y =  atom_positions_y((i-1)*81+1:i*81);
   planes(i).plane_y = planes(i).plane_y';
   % create a finer x vector for the x positions
   planes(i).fine_x = min_x:step_size:max_x;
   planes(i).fine_x = planes(i).fine_x';
end

%% plot the HAADF image with overlaid atomic positions
fig1=figure(1);
set(fig1,'Position',[0, 100, 2000, 300]);
imagesc(tem_image_x,tem_image_y,tem_image);
hold on
scatter(atom_positions_x,atom_positions_y,10,'MarkerEdgeColor',[0 .5 .5],...
    'LineWidth',1.5);
hold off
% set(gca,'YDir','normal')
% caxis([0 10])
colormap(gray);
set(gca,'Color',[1, 1, 1])
set(gca, 'LineWidth', 1)
set(gca, 'box','on')
set(gca,'Layer','top')
set(gca,'fontsize',21);
axis equal
set(gca,'TickLength',[0.0, 0.0])
xlabel('x (nm)','interpreter','latex');
ylabel('y (nm)','interpreter','latex');
c.Label.String = 'Intensity (a.u.)';
c.Label.Interpreter= 'latex';
c.FontSize = 21;
xlim([0 25]);
ylim([min(tem_image_y) max(tem_image_y)]);
set(gcf, 'Color', 'w');
fig1.InvertHardcopy = 'off';

%% fit a sine wave to each plane
% four parameters - a, b, c and d
sine_fit =  fittype(['a*sin(2*pi*(x+b)/c) + d']);

% define starting parameters, low and high limits
startCoeff = [0.35  20      19      10      ]; %[a b c d]
lowLimit   = [0.01  0.01    0.01    0.01    ];
highLimit  = [100   100     100     100     ];

% create a structure with all of the information of the fit
fit_info = struct;
% create structure with errors associated with the fit
fit_errors = struct;

x_colormap = [];
y_colormap = [];
curvature_colormap = [];

for i=1:6
    weights=0*planes(i).plane_y+1;
    % do not weigh atoms further than 20 nm from the origin - deviation
    % from sine wave behaviour
    weights(planes(i).plane_x>20)=0;
    % fit plane (i)
    fitResult = fit(planes(i).plane_x, planes(i).plane_y, sine_fit, 'Weights', weights,...
        'StartPoint', startCoeff, 'Lower', lowLimit, 'Upper',highLimit)
    var_names = coeffnames(fitResult);
    confidence_intervals = confint(fitResult);
    
    for j=1:4
        % calculate the errors associated with each variable
        fit_errors(i).(var_names{j,1}) = (confidence_intervals(2,j)-confidence_intervals(1,j))/2;
        % input fit results into fit_info structure
        fit_info(i).(var_names{j,1}) = fitResult.(var_names(j));
    end

    % calculate the curvature
    % z = z0sin(Ax)
    % curvature = d2z/dx2 = -A^2*z0*sin(Ax)
    fit_info(i).curvature = (2*pi/fit_info(i).c)^2*fit_info(i).a*1e9;
    % calculate the z-values of the fitted planes - at each x-position
    x = planes(i).fine_x;
    planes(i).fit = fit_info(i).a*sin(2*pi*(x+fit_info(i).b)/fit_info(i).c) +...
        fit_info(i).d;
    % calculate the curvature values at each x-position
    planes(i).curvature = (planes(i).fit-fit_info(i).d)...
        *(2*pi/fit_info(i).c)^2*1e9;
    
    % interpolate the curvature values to create the colormap
    if isempty(planes(i).X)
        [planes(i).X,planes(i).Y] = meshgrid(planes(i).fine_x,planes(i).fit);
%         f = scatteredInterpolant(planes(i).fine_x,planes(i).fit,planes(i).curvature);
        f = scatteredInterpolant(planes(i).fine_x,planes(i).fit,planes(i).curvature,'linear','none');
        planes(i).Z = f(planes(i).X,planes(i).Y);
        planes(i).x_vec = planes(i).X(1,:);
        planes(i).y_vec = planes(i).Y(:,1);
        planes(i).Z(isnan(planes(i).Z))=0;
    end
    
    % create x, y grid as well as interpolated curvature to make a
    % colormap
    x_colormap = [x_colormap ; planes(i).fine_x];
    y_colormap = [y_colormap ; planes(i).fit];
    curvature_colormap = [curvature_colormap ; planes(i).curvature];

    % plot fits with fit_info in each figure
    fig2 = figure(2);
    set(fig2, 'Position', [700 150 800 600]);
    plot(fitResult,planes(1).plane_x, planes(i).plane_y);
    
    dim = [0.5 0.5 0.3 0.3];
    a_str = ['a = ', num2str(fit_info(i).a)];
    b_str = ['b = ', num2str(fit_info(i).b)];
    c_str = ['c = ', num2str(fit_info(i).c)];
    d_str = ['d = ', num2str(fit_info(i).d)];
    curvature_str = ['maximum curvature = ', num2str(fit_info(i).curvature,'%.3e')];
    str = {'y = asin(2\pi(x+b)/c)+d',a_str,b_str,c_str,d_str, curvature_str};
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
    filename = ['Matlab_output/pto_sro_sublattice_01_sine_fit_0',num2str(i),'.png'];
    saveas(fig2,[path,filename]);
    close(fig2);
end

%% make curvature into a matrix
fine_x_colormap = min(x_colormap):step_size:max(x_colormap);
fine_y_colormap = min(y_colormap):step_size:max(y_colormap);

% interpolate curvature
[fine_X,fine_Y] = meshgrid(fine_x_colormap,fine_y_colormap);
f_colormap = scatteredInterpolant(x_colormap,y_colormap,curvature_colormap,'linear','none');
curvature_mesh = f_colormap(fine_X,fine_Y);

% make any NaN values equal to zero
curvature_mesh(isnan(curvature_mesh))=0;
% make X and Y vectors to make the plot easier
fine_X_vec = fine_X(1,:);
fine_Y_vec = fine_Y(:,1);

%% plot the sinusoidal fits
fig3=figure(3);
set(fig3,'Position',[0, 100, 2000, 300]);
imagesc(tem_image_x,tem_image_y,tem_image);
hold on
scatter(atom_positions_x,atom_positions_y,10,'MarkerEdgeColor',[0 .5 .5],...
    'LineWidth',1.5);
for i=1:6
   plot(planes(i).fine_x,planes(i).fit); 
end
hold off
% set(gca,'YDir','normal')
% caxis([-2e7 2e7])
colormap(gray);
% c = colorbar;
set(gca,'Color',[1, 1, 1])
set(gca, 'LineWidth', 1)
set(gca, 'box','on')
set(gca,'Layer','top')
set(gca,'fontsize',21);
axis equal
set(gca,'TickLength',[0.0, 0.0])
xlabel('x (nm)','interpreter','latex');
ylabel('y (nm)','interpreter','latex');
c.Label.String = 'Intensity (a.u.)';
c.Label.Interpreter= 'latex';
c.FontSize = 21;
xlim([0 25]);
ylim([min(tem_image_y) max(tem_image_y)]);
set(gcf, 'Color', 'w');
fig3.InvertHardcopy = 'off';

%% plot the curvature
map = brewermap(40,'*RdBu'); %brewermap package - red, white and blue colour scale
fig4=figure(4);
set(fig4,'Position',[0, 100, 2000, 300]);

imagesc(fine_X_vec,fine_Y_vec,curvature_mesh);

% set(gca,'YDir','normal')
caxis([-1.5e7 1.5e7])
colormap(map);
c = colorbar;
c.TickLength = 0.05;
set(gca,'Color',[1, 1, 1])
set(gca, 'LineWidth', 1)
set(gca, 'box','on')
set(gca,'Layer','top')
set(gca,'fontsize',21);
axis equal
% set(gca,'TickLength',[0.02, 0.02])
set(gca,'TickLength',[0.00, 0.00])
xlabel('x (nm)','interpreter','latex');
ylabel('y (nm)','interpreter','latex');
c.Label.String = 'Curvature (m$^{-1}$)';
c.Label.Interpreter= 'latex';
c.Label.Position = [4.5, 0];
c.FontSize = 21;
xlim([0 25]);
ylim([min(tem_image_y) max(tem_image_y)]);
set(gcf, 'Color', 'w');
fig4.InvertHardcopy = 'off';

%% plot the curvature - no colorbar
fig5=figure(5);
set(fig5,'Position',[0, 100, 2000, 300]);

imagesc(fine_X_vec,fine_Y_vec,curvature_mesh);

% set(gca,'YDir','normal')
caxis([-1.5e7 1.5e7])
colormap(map);
% c = colorbar;
set(gca,'Color',[1, 1, 1])
set(gca, 'LineWidth', 1)
set(gca, 'box','on')
set(gca,'Layer','top')
set(gca,'fontsize',21);
axis equal
% set(gca,'TickLength',[0.02, 0.02])
set(gca,'TickLength',[0.00, 0.00])
xlabel('x (nm)','interpreter','latex');
ylabel('y (nm)','interpreter','latex');
c.Label.String = 'Curvature (m$^{-1}$)';
c.Label.Interpreter= 'latex';
c.Label.Position = [4.5, 0];
c.FontSize = 21;
xlim([0 25]);
ylim([min(tem_image_y) max(tem_image_y)]);
set(gcf, 'Color', 'w');
fig5.InvertHardcopy = 'off';

%% save the figures
path = [path,'Matlab_output\'];
save('planes.mat','planes');
saveas(fig1,[path,'pto_sro_sublattice_01.png']);
saveas(fig3,[path,'pto_sro_sublattice_01_sine_fit.png']);
saveas(fig4,[path,'pto_sro_sublattice_01_curvature.png']);
