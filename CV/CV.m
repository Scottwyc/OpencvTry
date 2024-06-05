close all;
clear all;
Img=imread('D:\ÐÂÄ£ÐÍ\1.bmp');
Img=uint8(Img);
% Img=rgb2gray(Img);
% Img=imresize(Img,[256,256]);
Img=double(Img);
c0=1;
initialLSF = c0*ones(size(Img));
initialLSF(30:70,50:90) = -c0;
tic
phi_CV = CV_Func(Img,initialLSF);
toc
u=phi_CV;
figure(1);
imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal 
hold on;
contour(u,[0 0],'r');
% title('finial contour');
% print('-depsc','-r300',['finial contourCV' '.eps'])

