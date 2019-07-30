a=imread('coolCat.jpeg');
image(a)
 
figure
[x,y]=meshgrid(linspace(-3,3,201),linspace(-3,3,201));
k=normpdf(sqrt(x.^2+y.^2));
size(k);

imagesc(k)
image(k)
bw=mean(a,3);
imagesc(bw)
% >> size(bw)
% 
% ans =
% 
%    720   960
% 
 blurred_bw=conv2(bw,k,'same');
 imagesc(blurred_bw)
 
 imagesc(bw)
 
imagesc(blurred_bw)
 imagesc(bw)
 
 
 [x,y]=meshgrid(linspace(-9,9,201),linspace(-9,9,201));
 k=normpdf(sqrt(x.^2+y.^2));
 imagesc(k)
 
 %this creates the probe with the mesh grid abovd ofc
 %for finding vertical edges
 figure
 k=normpdf(sqrt(x.^2+y.^2)).*((sin(y)+sin(x)));
 imagesc(k)
 clear title
 title('plus')
 figure
 k=normpdf(sqrt(x.^2+y.^2)).*((sin(y)-sin(x)));
  imagesc(k)
 clear title
 title('minus')
% >> k=normpdf(sqrt(x.^2+y.^2)).*sin(x*2);
% >> imagesc(k)
% >> k=normpdf(sqrt(x.^2+y.^2)).*sin(x*6);
% >> imagesc(k)
figure
 blurred_bw=conv2(bw,k,'same');
 imagesc(blurred_bw)
% >> imagesc(bw)
% >> imagesc(blurred_bw)
% >> imagesc(blurred_bw.^2)
% >> imagesc(bw)
% >> 
% >> imagesc(k)
 
  k2=normpdf(sqrt(x.^2+y.^2)).*cos(x*6);
  imagesc(k)
 imagesc(k2)
  imagesc(blurred_bw.^2)
  blurred_bw2=conv2(bw,k2,'same');
  figure
  imagesc(blurred_bw2.^2)
  figure
 imagesc(blurred_bw2.^2+blurred_bw.^2)