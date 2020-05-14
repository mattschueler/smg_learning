clear; clc;
test = 'ThuFlex-200205-01';
filename = sprintf('verasonics/%s.mat',test);
m = matfile(filename);
id = m.ImgData;
id = id{1,1};
id = id(:,:,1,2:end-1);
id = reshape(id,[],size(id,4));
csvwrite(sprintf('vera_proc/%s.csv',test), id');