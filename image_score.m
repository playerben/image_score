function out = image_score(im)
fea = feature_extraction(im);
%% classifiy
fid = fopen('test_ind.txt','w');
for j = 1:size(fea,1)
    fprintf(fid,'%d ',j);
    for k = 1:size(fea,2)
        fprintf(fid,'%d:%f ',k,fea(j,k));
    end
    fprintf(fid,'\n');
end
fclose(fid);

system(['svm-scale -r range_class test_ind.txt >> test_ind_scaled']);
system(['svm-predict -b 1 test_ind_scaled model_class output_class']);
delete test_ind.txt test_ind_scaled

%% score 
fid = fopen('test_ind.txt','w');
for j = 1:size(fea,1)
    fprintf(fid,'%d ',j);
    for k = 1:size(fea,2)
        fprintf(fid,'%d:%f ',k,fea(j,k));
    end
    fprintf(fid,'\n');
end
fclose(fid);

system(['svm-scale -r range_td test_ind.txt >> test_ind_scaled']);
system(['svm-predict -b 1 test_ind_scaled model_td output_td']);
load output_td
td_score = output_td;
delete output_td test_ind_scaled

system(['svm-scale -r range_kl test_ind.txt >> test_ind_scaled']);
system(['svm-predict -b 1 test_ind_scaled model_kl output_kl']);
load output_kl
kl_score = output_kl;
delete output_kl test_ind_scaled

system(['svm-scale -r range_sj test_ind.txt >> test_ind_scaled']);
system(['svm-predict -b 1 test_ind_scaled model_sj output_sj']);
load output_sj
sj_score = output_sj;
delete output_sj test_ind_scaled
delete test_ind.txt 

fid = fopen('output_class','r');
fgetl(fid);
C = textscan(fid,'%f %f %f %f');
output = [C{1} C{2} C{3} C{4}];
fclose(fid);
probs = output(:,2:end);
scores  = [td_score kl_score sj_score];
out = sum(probs.*scores,2);
delete output_class
