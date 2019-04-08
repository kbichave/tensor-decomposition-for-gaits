%%
Data=Sensor120181211Accelerometer;
Data1=Data;
Data1{:,5:6}=Data{:,5:6}*-1;
[crossingsofinterest lengthchunk]=chopi(Data1, 4942,8615,0,-3.6)
%%
recovered2=tensormaking(Data1,crossingsofinterest,283,lengthchunk);
 %%
% sensor2_back_gyro=tensormaking(Sensor1Session220181031Gyroscope,crossingsofinterest,283,lengthchunk);
% KshitijLFWalk6=tensormaking(KshitijNormalWalkLF1Trial6,crossingsofinterest,283,lengthchunk)
%%
figure;
sensor2_back=recovered2
for i=1:80
    subplot(3,1,1);
    temp=zeros(1,283)
    temp(:,:)=sensor2_back(i,1,:);
    plot(temp);
    hold on;
    
    subplot(3,1,2);
    temp=zeros(1,283)
    temp(:,:)=sensor2_back(i,2,:);
    plot(temp);
    hold on;
    
    
    subplot(3,1,3);
    temp=zeros(1,283)
    temp(:,:)=sensor2_back(i,3,:);
    plot(temp);
    hold on;
end

%%
% figure;
% 
% for i=1:43
%     subplot(3,1,1);
%     temp=zeros(1,283)
%     temp(:,:)=sensor2_back_gyro(i,1,:);
%     plot(temp);
%     hold on;
%     
%     subplot(3,1,2);
%     temp=zeros(1,283)
%     temp(:,:)=sensor2_back_gyro(i,2,:);
%     plot(temp);
%     hold on;
%     
%     
%     subplot(3,1,3);
%     temp=zeros(1,283)
%     temp(:,:)=sensor2_back_gyro(i,3,:);
%     plot(temp);
%     hold on;
% end
    
    