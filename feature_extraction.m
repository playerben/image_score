function rep_vec = feature_extraction(im)
if(size(im,3)~=1)
    im = rgb2gray(im);
end

num_scales = 3; % somethings are hardcoded for this...please be careful when changing.
gam = 0.2:0.001:10;
r_gam = gamma(1./gam).*gamma(3./gam)./(gamma(2./gam)).^2;

[C S] = wavedec2(im,num_scales,'db9');
for p = 1:num_scales
    [horz_temp,vert_temp,diag_temp] = detcoef2('all',C,S,p) ;
    horz(p) = {[horz_temp(:)]};
    diag(p) = {[diag_temp(:)]};
    vert(p) = {[vert_temp(:)]};

    h_horz_curr  = cell2mat(horz(p));
    h_vert_curr  = cell2mat(vert(p));
    h_diag_curr  = cell2mat(diag(p));

    mu_horz(p) = mean(h_horz_curr);
    sigma_sq_horz(p)  = mean((h_horz_curr-mu_horz(p)).^2);
    E_horz = mean(abs(h_horz_curr-mu_horz(p)));
    rho_horz = sigma_sq_horz(p)/E_horz^2;
    [min_difference, array_position] = min(abs(rho_horz - r_gam));
    gam_horz(p) = gam(array_position);

    mu_vert(p) = mean(h_vert_curr);
    sigma_sq_vert(p)  = mean((h_vert_curr-mu_vert(p)).^2);
    E_vert = mean(abs(h_vert_curr-mu_vert(p)));
    rho_vert = sigma_sq_vert(p)/E_vert^2;
    [min_difference, array_position] = min(abs(rho_vert - r_gam));
    gam_vert(p) = gam(array_position);

    mu_diag(p) = mean(h_diag_curr);
    sigma_sq_diag(p)  = mean((h_diag_curr-mu_diag(p)).^2);
    E_diag = mean(abs(h_diag_curr-mu_diag(p)));
    rho_diag = sigma_sq_diag(p)/E_diag^2;
    [min_difference, array_position] = min(abs(rho_diag - r_gam));
    gam_diag(p) = gam(array_position);
end
rep_vec = [mu_horz mu_vert mu_diag sigma_sq_horz sigma_sq_vert sigma_sq_diag gam_horz gam_vert gam_diag];
rep_vec(:,1:9) = []; % remove the means...
end