function phi_CV = CV_Func(Img,u)

[nrow,ncol] = size(Img);

numIter = 5; timestep = 0.1;
lambda_1=1; lambda_2=1;
h = 1; epsilon=1; nu = 0.001*255*255;

figure; imagesc(Img,[0 255]); colormap(gray); hold on; contour(u,[0 0],'r');


% start level set evolution
for k=1:numIter
    u_new=u;
    u=EVOL_CV(Img, u_new, nu, lambda_1, lambda_2, timestep, epsilon, 1);   % update level set function
%     a(k)=norm(u_new-u,2)
%     if mod(k,1)==0
%         pause(.1);
%         imagesc(Img,[0 255]);colormap(gray);  axis off; axis equal;
%         hold on;
%         contour(u,[0 0],'r');
%         iterNum=[num2str(k), ' iterations'];
%         title(iterNum);
%         hold off;
%     end    
end;
phi_CV = u;