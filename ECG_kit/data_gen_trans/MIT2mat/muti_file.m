clc; clear all;
maindir = 'D:\aa_work\新学期项目及论文\心电图项目\数据集\MIT-BIH Arrhythmia Database\mat';
files  = dir( maindir );

for i = 1 : length( files )
    if( isequal( files( i ).name, '.' )||...
        isequal( files( i ).name, '..'))               
        continue;
    end
%     subdirpath = fullfile( maindir, subdir( i ).name, '*.dat' );

      datpath = fullfile( maindir, files( i ).name);
%         fid = fopen( datpath );
disp(datpath);
end