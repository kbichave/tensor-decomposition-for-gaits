%%
Data=Sensor2Session2Accelerometer20181022T16;
[crossingsofinterest lengthchunk]=chopi_back(Data, 3000,6000,-0.71)
%%
sensor2_back=tensormaking(Data,crossingsofinterest(2:51),60,lengthchunk(2:50));
%%
sensor2_back_gyro=tensormaking(Sensor2Session120181031Gyroscope,crossingsofinterest(2:51),60,lengthchunk(2:50));
% KshitijLFWalk6=tensormaking(KshitijNormalWalkLF1Trial6,crossingsofinterest,283,lengthchunk)
%%
figure;

for i=1:49
    subplot(3,1,1);
    temp=zeros(1,60)
    temp(:,:)=sensor2_back(i,1,:);
    plot(temp);
    hold on;
    
    subplot(3,1,2);
    temp=zeros(1,60)
    temp(:,:)=sensor2_back(i,2,:);
    plot(temp);
    hold on;
    
    
    subplot(3,1,3);
    temp=zeros(1,60)
    temp(:,:)=sensor2_back(i,3,:);
    plot(temp);
    hold on;
end

%%
figure;

for i=1:49
    subplot(3,1,1);
    temp=zeros(1,60)
    temp(:,:)=sensor2_back_gyro(i,1,:);
    plot(temp);
    hold on;
    
    subplot(3,1,2);
    temp=zeros(1,60)
    temp(:,:)=sensor2_back_gyro(i,2,:);
    plot(temp);
    hold on;
    
    
    subplot(3,1,3);
    temp=zeros(1,60)
    temp(:,:)=sensor2_back_gyro(i,3,:);
    plot(temp);
    hold on;
end
    
    